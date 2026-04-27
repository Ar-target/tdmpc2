# gym.Wrapper
# class Wrapper(
#     Env[WrapperObsType, WrapperActType],
#     Generic[WrapperObsType, WrapperActType, ObsType, ActType],
# ):
#     def __init__(self, env: Env[WrapperObsType, WrapperActType]):
#         """Wraps an environment to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.

#         Args:
#             env: The environment to wrap
#         """
#         self.env = env
#         assert isinstance(env, Env), (
#             f"Expected env to be a `gymnasium.Env` but got {type(env)}"
#         )

#         self._action_space: spaces.Space[WrapperActType] | None = None
#         self._observation_space: spaces.Space[WrapperObsType] | None = None
#         self._metadata: dict[str, Any] | None = None

#         self._cached_spec: EnvSpec | None = None

import gymnasium as gym
import numpy as np
import torch

from envs.wrappers.timeout import Timeout
from envs.tasks import offroad_driving
from collections import defaultdict, deque

# override
# 计算环境中所有观测项的总维度，并将其统一为一个扁平化的形状
def get_obs_shape(env):
	obs_shp = []
	for v in env.observation_spec().values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	return (int(np.sum(obs_shp)),)

# class ActionScaleWrapper(gym.ActionWrapper):
#     """
#     将底层环境的动作空间缩放到指定的范围 (默认是 [-1, 1])。
#     智能体输出 [-1, 1] 的动作，该 Wrapper 会自动将其还原为底层物理引擎真实的动作范围。
#     """
#     def __init__(self, env):
#         super().__init__(env)
        
#         # 记录底层环境真实的动作边界
#         self.env_low = self.env.action_space.low
#         self.env_high = self.env.action_space.high

#     def action(self, action):
#         """
#         在 env.step(action) 被调用前，这个函数会自动拦截并转换 action。
#         """
#         # 1. 安全裁剪：防止神经网络输出的动作由于浮点误差略微超出 [minimum, maximum]
#         action = np.clip(action, -1.0, 1.0)
        
#         # 2. 线性映射公式：将动作从 [-1, 1] 映射回真实的 [env_low, env_high]
#         # (action - min) / (max - min) 会得到一个 0 到 1 之间的比例
#         norm_action = (action + 1.0) / 2
        
#         # 根据比例计算真实的物理动作值
#         scaled_action = self.env_low + norm_action * (self.env_high - self.env_low)
        
#         return scaled_action

class BeamNGWrapper:
    def __init__(self, env):
        self.env = env

        # 状态与动作形状获取
        obs_shape = get_obs_shape(self.env)
        action_shape = self.env.action_spec().shape
        
        # 定义观测空间
        self.observation_space = gym.spaces.Box(
			low=np.full(obs_shape, -np.inf, dtype=np.float32),
			high=np.full(obs_shape, np.inf, dtype=np.float32),
			dtype=np.float32)
        
        # 定义动作空间
        self.action_space = gym.spaces.Box(
			low=np.full(action_shape, self.env.action_spec().minimum),
			high=np.full(action_shape, self.env.action_spec().maximum),
			dtype=env.action_spec().dtype)
        
        # 提取并存储环境定义的动作数据类型
        # self.action_spec_dtype = self.env.action_spec().dtype

    def step(self, action):
        # action = action.astype(self.action_spec_dtype)
        state_obs, reward, done, info = self.env.step(action)
        return state_obs, reward, done, info

    # override
    @property
    def unwrapped(self):
        return self.env

class Multimodal(gym.Wrapper):
    def __init__(self, env, num_frames=3):
        super().__init__(env)
        self._frames = deque([], maxlen=num_frames)
        
        # 1. 获取底层 BeamNGWrapper 扁平化后的物理向量空间
        state_space = self.env.observation_space
        
        # 2. 重新定义观测空间为一个字典 (Dict)
        self.observation_space = gym.spaces.Dict({
            'rgb': gym.spaces.Box(
                low=0.0, high=1.0, shape=(num_frames * 3, 64, 64), dtype=np.float32
            ),
            'state': state_space  
        })

    def step(self, action):
        state_obs, reward, done, info = self.env.step(action)
        obs = self._get_obs(state_obs)
        return obs, reward, done, info
    
    def reset(self):
        # 拿到底层重置后的物理状态向量
        state_obs = self.env.reset()
        obs = self._get_obs(state_obs, is_reset=True)
        return obs
    
    def _get_obs(self, state_obs, is_reset=False):
        frame = self.env.render().transpose(2, 0, 1) # 转换为 TD-MPC 期望的 CHW 格式
        img_tensor = frame.astype(np.float32) / 255.0 # 归一化到 [0, 1]
        
        # 如果是 reset，用第一帧填满整个双端队列
        num_frames = self._frames.maxlen if is_reset else 1
        for _ in range(num_frames):
            self._frames.append(img_tensor)
            
        # 拼接图像帧 (9, 64, 64)
        stacked_images = torch.from_numpy(np.concatenate(self._frames))
        
        # 将图像和底层的状态向量一起打包成字典返回
        return {
            'rgb': stacked_images,
            'state': state_obs  
        }

def make_env(cfg):
    task_name = cfg.task

    TASK_MAP = {
        'cruise': offroad_driving.cruise,
        'obstacle_avoidance': offroad_driving.obstacle_avoidance
    }
    if task_name not in TASK_MAP:
        raise ValueError(f'Unknown BeamNG task: {task_name}')
    
    env = TASK_MAP[task_name](cfg)
    env = Multimodal(env)
    env = Timeout(env, max_episode_steps=cfg.episode_length)
    
    return env