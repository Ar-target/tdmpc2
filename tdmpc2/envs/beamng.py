from collections import defaultdict
import gymnasium as gym
import numpy as np
import torch
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import State, Damage
from envs.wrappers.timeout import Timeout

# 状态维度扩大到15维: pos(3) + vel(3) + dir(3) + up(3) + rel_goal(3)
OBS_DIM = 15  
ACT_DIM = 3  # steering, throttle, brake

def get_obs_from_state(state: dict, goal_pos: np.ndarray) -> np.ndarray:
    pos       = np.array(state['pos'], dtype=np.float32)   # (3,)
    vel       = np.array(state['vel'], dtype=np.float32)   # (3,)
    direction = np.array(state['dir'], dtype=np.float32)   # (3,)
    up        = np.array(state['up'], dtype=np.float32)    # (3,) 向上向量，用于判断翻车
    rel_goal  = goal_pos - pos                             # (3,) 与终点的相对向量
    
    # 将所有信息拼接成一个15维的一维向量
    return np.concatenate([pos, vel, direction, up, rel_goal])


class BeamNGWrapper:

    def __init__(self, cfg):
        self.cfg = cfg

        self.bng = BeamNGpy(
            host=cfg.beamng_host,
            port=cfg.beamng_port,
            home=cfg.beamng_home
        )
        self.bng.open(launch=True)

        # 设定终点坐标 (示例：在起点Y轴前方200米处，请根据你的Utah地图实际情况修改)
        self.goal_pos = np.array(self.cfg.start_pos) + np.array([0.0, 200.0, 0.0])

        self._setup_scenario()

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low =np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.prev_dist = None # 记录上一帧距离终点的距离，用于计算奖励

    def _setup_scenario(self):
        self.scenario = Scenario(self.cfg.map, 'rl_scenario')
        self.vehicle  = Vehicle('ego', model=self.cfg.vehicle_model, licence='RL')
        
        # 附加伤害传感器用于碰撞/避障检测
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
        state = self.vehicle.state # 直接从 state 属性获取
        return torch.from_numpy(get_obs_from_state(state, self.goal_pos)).float()

    def reset(self) -> torch.Tensor:
        self.bng.restart_scenario()
        self.bng.pause()
        
        # 初始化上一帧的距离
        self.vehicle.sensors.poll()
        pos = np.array(self.vehicle.state['pos'], dtype=np.float32)
        self.prev_dist = np.linalg.norm(self.goal_pos - pos)
        
        return self._get_obs()

    def step(self, action: np.ndarray):
        steering = float(action[0])
        throttle = float(action[1])
        brake    = float(action[2])
        
        reward = 0.0
        done = False
        info = defaultdict(float)

        for _ in range(2):
            self.vehicle.control(
                steering=steering,
                throttle=throttle,
                brake=brake
            )
            self.bng.step(1)
            self.vehicle.sensors.poll()
            
            state = self.vehicle.state
            damage_data = self.vehicle.sensors['damage']
            
            pos = np.array(state['pos'], dtype=np.float32)
            up = np.array(state['up'], dtype=np.float32)
            dist_to_goal = np.linalg.norm(self.goal_pos - pos)
            
            # --- 1. 进度奖励 (向目标靠近) ---
            progress = self.prev_dist - dist_to_goal
            self.prev_dist = dist_to_goal
            reward += progress * 1.0  # 靠近给正奖励，远离给负奖励
            
            # --- 2. 避障惩罚 (基于车损) ---
            if damage_data['damage'] > 0:
                reward -= 150.0  # 发生碰撞，给付极大惩罚
                done = True
                break
                
            # --- 3. 防翻车惩罚 ---
            # 如果 up 向量的 Z 轴分量小于 0，说明车顶朝下了
            if up[2] < 0.0:
                reward -= 150.0
                done = True
                break
                
            # --- 4. 到达终点 ---
            if dist_to_goal < 5.0:  # 距离终点小于5米视为到达
                reward += 1000.0
                done = True
                break

        obs  = torch.from_numpy(get_obs_from_state(state, self.goal_pos)).float()
        reward = float(np.clip(reward, -500.0, 1000.0))
        return obs, reward, done, info

    def render(self, width=384, height=384, camera_id=None):
        raise NotImplementedError('rgb 模式请在 cfg 中配置 Camera 传感器。')

    def close(self):
        self.bng.close()

def make_env(cfg):
    if not cfg.task.startswith('beamng-'):
        raise ValueError(f'Not a BeamNG task: {cfg.task}')
    env = BeamNGWrapper(cfg)
    env = Timeout(env, max_episode_steps=cfg.episode_length) # 使用 cfg 中的配置
    return env