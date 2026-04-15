from collections import defaultdict
import gymnasium as gym
import numpy as np
import torch
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import State, Damage
from envs.wrappers.timeout import Timeout

# ============================================================
# 观测维度: vel(3) + dir(3) + up(3) + rel_goal_norm(3) + speed(1) + heading_dot(1) = 14维
# 去掉绝对坐标pos，防止模型过拟合起点位置
# ============================================================
OBS_DIM = 14
ACT_DIM = 3  # steering, throttle, brake

GOAL_DIST = 200.0  # 用于归一化rel_goal


def get_obs_from_state(state: dict, goal_pos: np.ndarray) -> np.ndarray:
    """
    构建14维观测向量：
      vel(3)           - 速度向量（包含方向和大小信息）
      dir(3)           - 车头朝向单位向量
      up(3)            - 车体up向量（用于检测翻车）
      rel_goal_norm(3) - 归一化后的相对目标向量（除以GOAL_DIST，量级约为[-1,1]）
      speed(1)         - 速度标量（显式提供，方便策略网络利用）
      heading_dot(1)   - 车头朝向与目标方向的点积（[-1,1]，衡量对齐程度）
    """
    pos       = np.array(state['pos'],  dtype=np.float32)
    vel       = np.array(state['vel'],  dtype=np.float32)
    direction = np.array(state['dir'],  dtype=np.float32)
    up        = np.array(state['up'],   dtype=np.float32)
    rel_goal  = goal_pos - pos

    speed       = np.linalg.norm(vel).reshape(1).astype(np.float32)
    goal_dir    = rel_goal / (np.linalg.norm(rel_goal) + 1e-6)
    heading_dot = np.dot(direction[:2], goal_dir[:2]).reshape(1).astype(np.float32)
    rel_goal_norm = (rel_goal / GOAL_DIST).astype(np.float32)

    return np.concatenate([vel, direction, up, rel_goal_norm, speed, heading_dot])


class BeamNGWrapper:

    def __init__(self, cfg):
        self.cfg = cfg

        self.bng = BeamNGpy(
            host=cfg.beamng_host,
            port=cfg.beamng_port,
            home=cfg.beamng_home
        )
        self.bng.open(launch=True)

        # 终点：起点Y轴前方200米
        self.goal_pos = np.array(self.cfg.start_pos, dtype=np.float32) + np.array([0.0, GOAL_DIST, 0.0])

        self._setup_scenario()

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low =np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.prev_dist    = None   # 上一帧距终点距离
        self._prev_damage = 0.0    # 上一帧累计车损（用增量判断碰撞）

    def _setup_scenario(self):
        self.scenario = Scenario(self.cfg.map, 'rl_scenario')
        self.vehicle  = Vehicle('ego', model=self.cfg.vehicle_model, licence='RL')

        self.damage_sensor = Damage()
        self.vehicle.sensors.attach('damage', self.damage_sensor)

        self.scenario.add_vehicle(
            self.vehicle,
            pos=tuple(self.cfg.start_pos),
            rot_quat=tuple(self.cfg.start_rot)
        )
        self.scenario.make(self.bng)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        self.bng.pause()

    @property
    def unwrapped(self):
        return self.bng

    def _get_obs(self) -> torch.Tensor:
        self.vehicle.sensors.poll()
        state = self.vehicle.state
        return torch.from_numpy(get_obs_from_state(state, self.goal_pos)).float()

    def reset(self) -> torch.Tensor:
        self.bng.restart_scenario()
        self.bng.pause()

        # 重置状态
        self._prev_damage = 0.0
        self.vehicle.sensors.poll()
        pos = np.array(self.vehicle.state['pos'], dtype=np.float32)
        self.prev_dist = np.linalg.norm(self.goal_pos - pos)

        return self._get_obs()

    def step(self, action: np.ndarray):
        steering = float(action[0])
        throttle = float(action[1])
        brake    = float(action[2])

        reward = 0.0
        done   = False
        info   = defaultdict(float)
        state  = None

        for _ in range(2):
            self.vehicle.control(
                steering=steering,
                throttle=throttle,
                brake=brake
            )
            self.bng.step(1)
            self.vehicle.sensors.poll()

            state       = self.vehicle.state
            damage_data = self.vehicle.sensors['damage']

            pos          = np.array(state['pos'], dtype=np.float32)
            up           = np.array(state['up'],  dtype=np.float32)
            direction    = np.array(state['dir'], dtype=np.float32)
            vel          = np.array(state['vel'], dtype=np.float32)
            dist_to_goal = np.linalg.norm(self.goal_pos - pos)

            # --- 1. 进度奖励：靠近目标给正奖励，远离给负奖励 ---
            progress = self.prev_dist - dist_to_goal
            self.prev_dist = dist_to_goal
            reward += progress * 1.0

            # --- 2. 朝向奖励：车头与目标方向对齐 ---
            goal_dir    = (self.goal_pos - pos) / (dist_to_goal + 1e-6)
            heading_dot = float(np.dot(direction[:2], goal_dir[:2]))
            reward += heading_dot * 0.5  # 朝向完全对齐最多每sub-step +0.5

            # --- 3. 存活奖励：鼓励智能体尽量不要"早死" ---
            reward += 0.05

            # --- 4. 碰撞惩罚：用增量检测，避免累计值导致每步都惩罚 ---
            current_damage = float(damage_data['damage'])
            delta_damage   = current_damage - self._prev_damage
            self._prev_damage = current_damage
            if delta_damage > 50.0:  # 阈值：只有显著碰撞才终止
                reward -= 150.0
                done = True
                info['termination'] = 'collision'
                break

            # --- 5. 防翻车惩罚：up向量Z分量<0表示车顶朝下 ---
            if up[2] < 0.0:
                reward -= 150.0
                done = True
                info['termination'] = 'rollover'
                break

            # --- 6. 到达终点 ---
            if dist_to_goal < 5.0:
                reward += 1000.0
                done = True
                info['termination'] = 'success'
                break

        obs    = torch.from_numpy(get_obs_from_state(state, self.goal_pos)).float()
        reward = float(np.clip(reward, -500.0, 1100.0))  # 上限略高于1000以保留终点奖励精度
        info['dist_to_goal'] = float(dist_to_goal) if dist_to_goal is not None else 0.0
        info['success'] = (dist_to_goal < 5.0)
        info['terminated'] = done
        return obs, reward, done, info

    def render(self, width=384, height=384, camera_id=None):
        raise NotImplementedError('rgb 模式请在 cfg 中配置 Camera 传感器。')

    def close(self):
        self.bng.close()


def make_env(cfg):
    if not cfg.task.startswith('beamng-'):
        raise ValueError(f'Not a BeamNG task: {cfg.task}')
    env = BeamNGWrapper(cfg)
    env = Timeout(env, max_episode_steps=cfg.episode_length)
    return env