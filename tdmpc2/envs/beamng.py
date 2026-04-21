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

# 动作维度：steering, acc_pedal (目标加速度踏板：正为油门，负为刹车)
ACT_DIM = 2


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
# OU 噪声：用于 seed 阶段生成时间相关的平滑随机动作
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

    # seed 阶段的子策略列表
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

        # ── 2. 追踪进度 ──────────────────────────────────────────
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
        self.prev_dist    = None   
        self.current_step_count = 0
        self.stuck_counter = 0

        # ── 5. seed 阶段探索辅助 ──────────────────────────────────────────
        self._ou_noise        = OUNoise(ACT_DIM)
        self._explore_mode    = 'straight'
        self._explore_counter = 0

        # ── 6. 空间定义 (2D 动作空间) ────────────────────────────
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low =np.array([-1.0, -1.0], dtype=np.float32), 
            high=np.array([ 1.0,  1.0], dtype=np.float32), 
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
        self._explore_counter += 1
        if self._explore_counter % self.EXPLORE_SWITCH_STEPS == 0:
            self._explore_mode = np.random.choice(self._EXPLORE_MODES)

        mode_actions = {
            'straight'  : np.array([ 0.0,  0.8], dtype=np.float32),
            'turn_left' : np.array([-0.4,  0.6], dtype=np.float32),
            'turn_right': np.array([ 0.4,  0.6], dtype=np.float32),
            'brake'     : np.array([ 0.0, -0.8], dtype=np.float32),
        }
        base  = mode_actions[self._explore_mode]
        noise = self._ou_noise.sample() * np.array([0.2, 0.1], dtype=np.float32)
        action = np.clip(base + noise,
                         self.action_space.low,
                         self.action_space.high)
        return action

    def reset(self) -> torch.Tensor:
        self.bng.restart_scenario()
        self.bng.pause()

        # 强制挂入 1 档（越野推荐）或 D 档，踩死刹车，等待齿轮咬合
        self.vehicle.control(gear=1, throttle=0.0, brake=1.0, parkingbrake=0.0)
        self.bng.step(60) 

        self._prev_damage       = 0.0
        self.current_wp_index   = 1
        self._ou_noise.reset()
        self._explore_counter   = 0
        self._explore_mode      = 'straight'
        self.current_step_count = 0  
        self.stuck_counter      = 0

        self.vehicle.sensors.poll()
        pos             = np.array(self.vehicle.state['pos'], dtype=np.float32)
        target, _       = self._get_current_targets()
        self.prev_dist  = float(np.linalg.norm(target - pos))

        return self._get_obs()

    def step(self, action: np.ndarray):
        self.current_step_count += 1
        
        # ── 1D 踏板解析 ──
        steering = float(np.clip(action[0], -1.0, 1.0))
        pedal    = float(np.clip(action[1], -1.0, 1.0))

        if pedal > 0:
            throttle = pedal
            brake    = 0.0
        else:
            throttle = 0.0
            brake    = -pedal 

        reward = 0.0
        done   = False
        info   = {}

        V_REF = 20.0 

        # ── 10Hz 控制频率 (0.1秒物理步长) ──
        # 有效过滤探索早期的纯随机高频噪声
        self.vehicle.control(steering=steering, throttle=throttle, brake=brake)
        self.bng.step(50)  
        self.vehicle.sensors.poll()

        state       = self.vehicle.state
        damage_data = self.vehicle.sensors['damage']
        pos         = np.array(state['pos'], dtype=np.float32)
        vel         = np.array(state['vel'], dtype=np.float32)
        direction   = np.array(state['dir'], dtype=np.float32)
        up          = np.array(state['up'],  dtype=np.float32)
        current_speed = np.linalg.norm(vel)

        target, _    = self._get_current_targets()
        rel_goal     = target - pos
        dist_to_goal = float(np.linalg.norm(rel_goal))

        stuck_flag = " [STUCK WARNING]" if self.stuck_counter > 5 else ""
        print(f"Step: {self.current_step_count:3d} | "
              f"Steer: {steering:5.2f} | "
              f"Thr: {throttle:4.2f} | "
              f"Brk: {brake:4.2f} | "
              f"Speed: {current_speed:5.2f} m/s | "
              f"Dist: {dist_to_goal:5.1f} m | "
              f"WP: {self.current_wp_index}{stuck_flag}")

        if dist_to_goal < self.wp_radius:
            self.current_wp_index += 1
            reward += 1.0 

            if self.current_wp_index >= self.num_waypoints:
                reward += 5.0 
                done = True
                info['termination'] = 'success'
            else:
                target, _    = self._get_current_targets()
                rel_goal     = target - pos
                dist_to_goal = float(np.linalg.norm(rel_goal))

        goal_dir = rel_goal / (dist_to_goal + 1e-6)
        proj_vel   = float(np.dot(vel, goal_dir))
        
        # A. 速度奖励
        reward_vel = 0.6 * float(np.tanh(max(0.0, proj_vel) / V_REF))

        # C. 存活奖励
        reward_survival = 0.1

        reward += reward_vel + reward_survival

        # E. 低速惩罚
        if proj_vel < 2.0 and self.current_step_count > 25: # 25步 = 2.5秒免罚
            reward -= 0.2

        # ── 移除倒车惩罚，鼓励智能体自行倒车脱困 ──
        # if proj_vel < -0.5: 
        #     reward -= 0.5

        # F. 放宽的碰撞判定
        current_damage = float(damage_data['damage'])
        delta_damage   = current_damage - self._prev_damage
        self._prev_damage = current_damage

        if delta_damage > 0:
            # 阈值提高到 5000，防止原地轰油门或轻微托底导致回合结束
            if delta_damage > 5000.0:
                reward -= 1.0             
                done   = True
                info['termination'] = 'collision'
            else:
                reward -= 0.05 # 仅给予微小惩罚            
                info['minor_hit'] = True

        # G. 翻车判定
        if up[2] < 0.0:
            reward = -1.0
            done   = True
            info['termination'] = 'rollover'
            
        # ── H. 强制防卡死保护 ──
        # 只要车停了（无论踩不踩刹车），统统算作卡死累计
        if self.current_step_count > 25 and current_speed < 0.5:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # 连续 20 步 (物理时间 2 秒) 卡在原地，重罚并结束
        if self.stuck_counter > 20:
            reward -= 2.0  
            done = True
            info['termination'] = 'stuck'

        self.prev_dist = dist_to_goal

        obs    = self._get_obs()
        reward = float(np.clip(reward, -2.5, 5.0)) # 放宽 reward 下限裁剪以容纳重罚

        info['dist_to_goal'] = float(dist_to_goal)
        info['current_wp']   = self.current_wp_index
        info['success']      = (self.current_wp_index >= self.num_waypoints)
        info['speed']        = float(current_speed)
        
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