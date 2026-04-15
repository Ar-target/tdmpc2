import json
import os
from collections import defaultdict
import gymnasium as gym
import numpy as np
import torch
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage
from envs.wrappers.timeout import Timeout

# 状态维度: pos(3) + vel(3) + dir(3) + up(3) + rel_goal(3) + next_rel_goal(3)
OBS_DIM = 18

# 动作维度：steering, throttle, brake
ACT_DIM = 3


def get_obs_from_state(
    state: dict,
    current_goal: np.ndarray,
    next_goal: np.ndarray,
) -> np.ndarray:
    pos           = np.array(state['pos'], dtype=np.float32)   # (3,)
    vel           = np.array(state['vel'], dtype=np.float32)   # (3,)
    direction     = np.array(state['dir'], dtype=np.float32)   # (3,)
    up            = np.array(state['up'],  dtype=np.float32)   # (3,) 向上向量，用于判断翻车
    rel_goal      = current_goal - pos                          # (3,) 当前航点相对向量
    next_rel_goal = next_goal    - pos                          # (3,) 前瞻航点相对向量
    return np.concatenate([pos, vel, direction, up, rel_goal, next_rel_goal])


# ---------------------------------------------------------------------------
# OU 噪声：用于 seed 阶段生成时间相关的平滑随机动作，比纯均匀随机更接近真实驾驶
# ---------------------------------------------------------------------------
class OUNoise:
    def __init__(self, action_dim: int, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.2):
        self.mu    = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(action_dim, dtype=np.float32)

    def reset(self):
        self.state = np.zeros_like(self.state)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(len(self.state)).astype(np.float32)
        self.state += dx
        return self.state.copy()


class BeamNGWrapper:

    # seed 阶段的子策略列表，每隔 EXPLORE_SWITCH_STEPS 步随机切换一次
    _EXPLORE_MODES       = ['straight', 'turn_left', 'turn_right', 'brake']
    EXPLORE_SWITCH_STEPS = 20

    def __init__(self, cfg):
        self.cfg = cfg

        # ── 1. 加载越野航点 JSON ──────────────────────────────────────────
        wp_file = os.path.join(
            os.path.dirname(__file__), '../data/utah_offroad_waypoints.json'
        )
        with open(wp_file, 'r') as f:
            wp_data = json.load(f)

        self.waypoints = [
            np.array([wp['x'], wp['y'], wp['z']], dtype=np.float32)
            for wp in wp_data['waypoints']
        ]
        self.wp_radius     = wp_data.get('waypoint_reach_radius', 4.0)
        self.num_waypoints = len(self.waypoints)

        # ── 2. 追踪进度（车辆初始生成在 WP 0，目标直接设为 WP 1）──────────
        self.current_wp_index = 1

        # ── 3. 启动 BeamNG ───────────────────────────────────────────────
        self.bng = BeamNGpy(
            host=cfg.beamng_host,
            port=cfg.beamng_port,
            home=cfg.beamng_home,
        )
        self.bng.open(launch=True)
        self._setup_scenario()

        # ── 4. 状态追踪变量 ───────────────────────────────────────────────
        self._prev_damage = 0.0
        self.prev_dist    = None   # 上一帧到当前目标航点的距离

        # ── 5. seed 阶段探索辅助 ──────────────────────────────────────────
        self._ou_noise        = OUNoise(ACT_DIM)
        self._explore_mode    = 'straight'
        self._explore_counter = 0

        # ── 6. 空间定义 ───────────────────────────────────────────────────
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low =np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # -----------------------------------------------------------------------
    # 内部工具
    # -----------------------------------------------------------------------

    def _setup_scenario(self):
        self.scenario = Scenario(self.cfg.map, 'rl_scenario')
        self.vehicle  = Vehicle('ego', model=self.cfg.vehicle_model, licence='RL')

        self.damage_sensor = Damage()
        self.vehicle.sensors.attach('damage', self.damage_sensor)

        self.scenario.add_vehicle(
            self.vehicle,
            pos=tuple(self.cfg.start_pos),
            rot_quat=tuple(self.cfg.start_rot),
        )
        self.scenario.make(self.bng)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        self.bng.pause()

    def _get_current_targets(self):
        """返回 (当前目标航点, 前瞻航点)"""
        target   = self.waypoints[self.current_wp_index]
        next_idx = min(self.current_wp_index + 1, self.num_waypoints - 1)
        return target, self.waypoints[next_idx]

    def _get_obs(self) -> torch.Tensor:
        self.vehicle.sensors.poll()
        state               = self.vehicle.state
        target, next_target = self._get_current_targets()
        obs_array           = get_obs_from_state(state, target, next_target)
        return torch.from_numpy(obs_array).float()

    # -----------------------------------------------------------------------
    # 公开接口
    # -----------------------------------------------------------------------

    @property
    def unwrapped(self):
        return self.bng

    def seed_action(self) -> np.ndarray:
        """
        seed 阶段专用的结构化随机动作：
        每 EXPLORE_SWITCH_STEPS 步切换一次子策略，叠加 OU 噪声保证平滑多样性。
        保证油门偏大，车辆真正能跑起来，覆盖更多状态空间。
        """
        self._explore_counter += 1
        if self._explore_counter % self.EXPLORE_SWITCH_STEPS == 0:
            self._explore_mode = np.random.choice(self._EXPLORE_MODES)

        mode_actions = {
            'straight'  : np.array([ 0.0, 0.8, 0.0], dtype=np.float32),
            'turn_left' : np.array([-0.4, 0.6, 0.0], dtype=np.float32),
            'turn_right': np.array([ 0.4, 0.6, 0.0], dtype=np.float32),
            'brake'     : np.array([ 0.0, 0.0, 0.8], dtype=np.float32),
        }
        base  = mode_actions[self._explore_mode]
        noise = self._ou_noise.sample() * np.array([0.2, 0.1, 0.05], dtype=np.float32)
        action = np.clip(base + noise,
                         self.action_space.low,
                         self.action_space.high)
        return action

    def reset(self) -> torch.Tensor:
        self.bng.restart_scenario()
        self.bng.pause()

        # 重置所有状态追踪变量
        self._prev_damage     = 0.0
        self.current_wp_index = 1
        self._ou_noise.reset()
        self._explore_counter = 0
        self._explore_mode    = 'straight'

        # 初始化 prev_dist，避免第一步 step() 出现 None - float 报错
        self.vehicle.sensors.poll()
        pos             = np.array(self.vehicle.state['pos'], dtype=np.float32)
        target, _       = self._get_current_targets()
        self.prev_dist  = float(np.linalg.norm(target - pos))

        return self._get_obs()

    def step(self, action: np.ndarray):
        # ── 动作 clip，防止越界传入 BeamNG ──────────────────────────────
        steering = float(np.clip(action[0], -1.0,  1.0))
        raw_throttle = float(np.clip(action[1],  0.0,  1.0))
        raw_brake    = float(np.clip(action[2],  0.0,  1.0))

        if raw_throttle > raw_brake:
            throttle, brake = raw_throttle, 0.0
        else:
            throttle, brake = 0.0, raw_brake

        reward = 0.0
        done   = False
        info   = defaultdict(float)

        V_REF = 20.0  # 参考速度（m/s），用于速度奖励归一化

        # 同一个动作连续执行 2 个仿真步
        for _ in range(2):
            self.vehicle.control(steering=steering, throttle=throttle, brake=brake)
            self.bng.step(1)
            self.vehicle.sensors.poll()

            state       = self.vehicle.state
            damage_data = self.vehicle.sensors['damage']
            pos         = np.array(state['pos'], dtype=np.float32)
            vel         = np.array(state['vel'], dtype=np.float32)
            direction   = np.array(state['dir'], dtype=np.float32)
            up          = np.array(state['up'],  dtype=np.float32)
            current_speed = np.linalg.norm(vel)

            # ── 1. 航点距离与通过判定 ─────────────────────────────────────
            target, _    = self._get_current_targets()
            rel_goal     = target - pos
            dist_to_goal = float(np.linalg.norm(rel_goal))

            if dist_to_goal < self.wp_radius:
                self.current_wp_index += 1
                reward += 1.0  # 阶段性通过奖励

                if self.current_wp_index >= self.num_waypoints:
                    reward += 5.0  # 全程完成大奖
                    done = True
                    info['termination'] = 'success'
                    break

                # 更新到下一个航点，保证后续奖励计算连贯
                target, _    = self._get_current_targets()
                rel_goal     = target - pos
                dist_to_goal = float(np.linalg.norm(rel_goal))

            goal_dir = rel_goal / (dist_to_goal + 1e-6)

            # ── 2. 朝向计算（后续多处复用）───────────────────────────────
            heading_dot = float(np.dot(direction[:2], goal_dir[:2]))

            # ── A. 速度奖励：鼓励朝目标方向提速，tanh 保证数值有界 ────────
            proj_vel   = float(np.dot(vel, goal_dir))
            reward_vel = 0.6 * float(np.tanh(max(0.0, proj_vel) / V_REF))

            # ── B. 朝向奖励：权重从 0.3 提升到 0.5，压制倒车行为 ──────────
            reward_heading = max(0.0, heading_dot) * 0.5

            # ── C. 存活奖励 ────────────────────────────────────────────────
            reward_survival = 0.1

            reward += reward_vel + reward_heading + reward_survival

            # ── D. 进度奖励：方向正确时权重更高（修复原代码重复叠加 bug）──
            progress = self.prev_dist - dist_to_goal
            if heading_dot > 0.5:
                reward += progress * 2.0   # 方向正确，全额进度奖励
            else:
                reward += progress * 0.2   # 方向错误，大幅削减进度奖励

            # ── E. 惩罚项 ─────────────────────────────────────────────────
            # 同时踩油门和刹车
            if throttle > 0.1 and brake > 0.1:
                reward -= 0.05 * (throttle + brake)

            # 车速过低（< 3.6 km/h）：抵消存活奖励，惩罚原地徘徊
            if proj_vel < 1.0:
                reward -= 0.1

            # 倒车惩罚：速度方向与目标方向相反
            if proj_vel < 0.0:
                reward -= 0.2

            # ── F. 碰撞判定 ────────────────────────────────────────────────
            current_damage = float(damage_data['damage'])
            delta_damage   = current_damage - self._prev_damage
            self._prev_damage = current_damage

            if delta_damage > 0:
                if delta_damage < 50.0:
                    reward -= 0.2             # 轻微擦碰，扣分但不中断
                    info['minor_hit'] = True
                else:
                    reward = -1.0             # 严重碰撞，终止本 episode
                    done   = True
                    info['termination'] = 'collision'
                    break

            # ── G. 翻车判定 ────────────────────────────────────────────────
            if up[2] < 0.0:
                reward = -1.0
                done   = True
                info['termination'] = 'rollover'
                break

            # 更新 prev_dist（放在循环内，保证每个子步都能正确计算进度）
            self.prev_dist = dist_to_goal

        obs    = self._get_obs()
        reward = float(np.clip(reward, -1.0, 5.0))

        info['dist_to_goal'] = float(dist_to_goal)
        info['current_wp']   = self.current_wp_index
        info['success']      = (self.current_wp_index >= self.num_waypoints)
        info['speed'] = float(current_speed)

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