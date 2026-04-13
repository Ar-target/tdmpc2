from collections import defaultdict
import gymnasium as gym
import numpy as np
import torch
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import State
from envs.wrappers.timeout import Timeout


OBS_DIM = 9  # pos(3) + vel(3) + dir(3)
ACT_DIM = 3   # steering, throttle, brake


def get_obs_from_state(state: dict) -> np.ndarray:
    pos      = np.array(state['pos'], dtype=np.float32)   # (3,)
    vel      = np.array(state['vel'], dtype=np.float32)   # (3,)
    direction= np.array(state['dir'], dtype=np.float32)   # (3,)
    return np.concatenate([pos, vel, direction])           # (9,)


class BeamNGWrapper:

    def __init__(self, cfg):
        self.cfg = cfg

        self.bng = BeamNGpy(
            host=cfg.beamng_host,
            port=cfg.beamng_port,
            home=cfg.beamng_home
        )
        self.bng.open(launch=True)

        self._setup_scenario()

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low =np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _setup_scenario(self):
        self.scenario = Scenario(self.cfg.map, 'rl_scenario')
        self.vehicle  = Vehicle('ego', model=self.cfg.vehicle_model, licence='RL')
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
        state = self.vehicle.sensors['state'].data
        print("state keys:", list(state.keys()))  # 临时调试
        return torch.from_numpy(get_obs_from_state(state))

    def _compute_reward(self, state: dict) -> float:
        vel = np.array(state['vel'], dtype=np.float32)
        fwd = np.array(state['dir'], dtype=np.float32)
        return float(np.dot(vel, fwd))

    def reset(self) -> torch.Tensor:
        self.bng.restart_scenario()
        self.bng.pause()
        return self._get_obs()

    def step(self, action: np.ndarray):
        steering = float(action[0])
        throttle = float(action[1])
        brake    = float(action[2])
        reward   = 0.0

        for _ in range(2):
            self.vehicle.control(
                steering=steering,
                throttle=throttle,
                brake=brake
            )
            self.bng.step(1)
            self.vehicle.poll_sensors()
            state = self.vehicle.sensors['state'].data
            reward += self._compute_reward(state)

        obs  = torch.from_numpy(get_obs_from_state(state))
        done = False
        info = defaultdict(float)
        return obs, reward, done, info

    def render(self, width=384, height=384, camera_id=None):
        raise NotImplementedError('rgb 模式请在 cfg 中配置 Camera 传感器。')

    def close(self):
        self.bng.close()


def make_env(cfg):
    if not cfg.task.startswith('beamng-'):
        raise ValueError(f'Not a BeamNG task: {cfg.task}')
    env = BeamNGWrapper(cfg)
    env = Timeout(env, max_episode_steps=500)
    return env