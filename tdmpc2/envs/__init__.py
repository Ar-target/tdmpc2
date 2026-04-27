from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

# try:
# 	from envs.beamng import make_env as make_beamng_env
# except:
# 	make_beamng_env = missing_dependencies

from envs.beamng import make_env as make_beamng_env

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env


def make_env(cfg):
	gym.logger.set_level(40)
	if cfg.multitask:
		env = make_multitask_env(cfg)
	else:
		env = make_beamng_env(cfg)
		env = TensorWrapper(env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	# cfg.seed_steps = max(1000, 5*cfg.episode_length)
	cfg.seed_steps = 500
	return env
