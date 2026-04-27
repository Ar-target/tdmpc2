import json
import gymnasium as gym
import numpy as np
import torch
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics
# from beamngpy.sensors import Damage
from beamngpy.sensors import Camera
# from beamngpy.sensors import Radar
from beamngpy.sensors import AdvancedIMU

def cruise(cfg, environment_kwargs=None):
    physics = BeamNGPhysics(cfg) # 启动物理引擎
    task = OffroadDrivingTask(
        task='cruise', 
        waypoint_file=cfg.waypoint_file, # 支持从 cfg 动态传入不同地图的路径
        wp_radius=4.0, 
        move_speed=15.0
    ) # 设置任务
    environment_kwargs = environment_kwargs or {} # 额外关键词参数
    return BeamNGEnvironment(physics, task, **environment_kwargs) # 生成训练环境

def obstacle_avoidance(cfg, environment_kwargs=None):
    """
    避障任务：在向目标航点行驶的过程中（类似竞速），避开预设的障碍物区域。
    """
    physics = BeamNGPhysics(cfg)
    
    # 模拟静态障碍物的坐标 (后期可替换为动态读取)
    mock_obstacles = [
        np.array([-10.0, 50.0, 0.0], dtype=np.float32),
        np.array([ 20.0, 120.0, 0.0], dtype=np.float32),
        np.array([-5.0, 200.0, 0.0], dtype=np.float32)
    ]
    
    task = OffroadDrivingTask(
        goal='obstacle_avoidance', 
        wp_radius=4.0, 
        obstacles=mock_obstacles, 
        obs_radius=3.0 # 障碍物碰撞判定半径
    )
    environment_kwargs = environment_kwargs or {}
    return BeamNGEnvironment(physics, task, **environment_kwargs)


# ─── 物理引擎层 ─────────────────────────────────────────────────────────
class BeamNGPhysics:
    def __init__(self, cfg):
        self.cfg = cfg

        # 启动仿真器
        self.bng = BeamNGpy(host=cfg.beamng_host, port=cfg.beamng_port, home=cfg.beamng_home)
        self.bng.open(launch=True)
        self.bng.set_deterministic()
        self.bng.set_steps_per_second(self.cfg.steps_per_second)
        
        # 设置场景
        self.scenario = Scenario(self.cfg.map, 'rl_scenario')
        self._add_waypoints()
        if self.cfg.vehicle_start_pos == 'auto':
            first_wp = self.waypoints[0]
            self.vehicle_start_pos = (first_wp[0], first_wp[1], first_wp[2] + 0.5) # 关键：Z轴加 0.5 米，让车辆“空降”落地，防止卡在地下
        else:
            self.vehicle_start_pos = tuple(self.cfg.vehicle_start_pos)
        self.vehicle = Vehicle('ego', model=self.cfg.vehicle_model, licence='RL')
        self.scenario.add_vehicle(self.vehicle, pos=self.vehicle_start_pos, rot_quat=tuple(self.cfg.vehicle_start_rot))
        # self._add_npcs()
        self.scenario.make(self.bng)

        # 加载、启动场景
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()

        # # npc 巡逻逻辑
        # for npc in self.npcs:
        #     npc.ai.set_mode('traffic') # 自主巡逻
        
        # 运行 10 帧以调整初始状态
        self.bng.step(10)
        self._add_sensors()

        # 暂停场景，等待接管
        self.bng.pause()
        
    def _add_sensors(self):
        # 电气传感器：获取档位、转速、油门、刹车、方向盘转角、车轮线速度
        self.electrics = Electrics()
        self.vehicle.attach_sensor('electrics', self.electrics)

        # 碰撞传感器：获取车体各部位的形变和受损程度
        # self.damage = Damage()
        # self.vehicle.attach_sensor('damage', self.damage)

        # IMU：获取高精度的真实加速度、角速度
        self.imu = AdvancedIMU('ego_imu', self.bng, self.vehicle, 
            pos=(0, 0, 1.0),      # 安装在车辆质心附近
            dir=(0, -1, 0),       # 朝向正前方
            up=(0, 0, 1)          # Z轴朝上
        )

        # [RGB-D 相机]：输出彩色图像和深度图
        self.camera = Camera('front_cam', self.bng, self.vehicle,
            pos=(0, -0.5, 1.5),     # 安装在挡风玻璃或车顶上方
            dir=(0, -1, 0),         # 朝向车头正前方
            field_of_view_y=70,     # FOV 视野大小
            resolution=(64, 64),  # 分辨率（RL 推荐降采样以加快训练）
            is_render_colours=True,  # 开启彩色渲染
            is_render_depth=False    # 开启深度图渲染
        )

        # 3D 激光雷达 (LiDAR)：输出精确的 3D 点云
        # self.lidar = Lidar('roof_lidar', self.bng, self.vehicle,
        #     pos=(0, 0, 1.8),        # 安装在车顶最高处
        #     dir=(0, -1, 0),
        #     vertical_resolution=64, # 64 线雷达
        #     vertical_angle=26.9,    # 垂直视场角
        #     horizontal_angle=360,   # 水平 360 度扫描
        #     max_distance=120.0,     # 最远探测距离(米)
        #     is_visualized=True      # 在游戏中显示激光束（方便调试，训练时建议关掉）
        # )

        # 毫米波雷达 (Radar)：输出前方障碍物的距离和相对速度
        # self.radar = Radar('front_radar', self.bng, self.vehicle,
        #     pos=(0, -2.0, 0.5),     # 安装在前保险杠
        #     dir=(0, -1, 0),
        #     field_of_view_y=10,     # 垂直范围较窄
        #     field_of_view_x=60,     # 水平范围较宽
        #     range_min=0.1,
        #     range_max=150.0,
        #     is_visualized=False
        # )
    
    def _add_waypoints(self):
        file_path = self.cfg.waypoint_file
        self.waypoints = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        raw_waypoints = data.get('waypoints', [])
        self.waypoints = [
            (float(wp['x']), float(wp['y']), float(wp['z'])) 
            for wp in raw_waypoints
        ]
        self.scenario.add_checkpoints(
            positions=self.waypoints,       
            scales=[(4,4,4)] * len(self.waypoints),
            ids=[f'wp_{i}' for i in range(len(self.waypoints))]
        )
    
    # def _add_npcs(self):
    #     self.npcs = []
    #     start_x, start_y, start_z = self.vehicle_start_pos # 放在 ego 起始点旁边 10 米处测试
        
    #     npc_vehicle_positions = [
    #         (start_x + 10, start_y, start_z + 0.5),
    #         (start_x - 10, start_y, start_z + 0.5)
    #     ]
        
    #     for i, pos in enumerate(npc_vehicle_positions):
    #         npc = Vehicle(f'npc_{i}', model='pickup', licence='TEST')
    #         self.scenario.add_vehicle(npc, pos=pos, rot_quat=(0, 0, 0, 1))
    #         self.npcs.append(npc)
    
    def reset(self):
        self.vehicle.recover()

        self.bng.vehicles.teleport(
            self.vehicle,
            pos=self.vehicle_start_pos,
            rot_quat=tuple(self.cfg.vehicle_start_rot),
            reset=True
        )

        self.vehicle.queue_lua_command('controller.getMainController().setIgnitionLevel(2)')
        self.vehicle.set_shift_mode('arcade')
        self.vehicle.control(throttle=0.0, brake=0.0, parkingbrake=0.0)
        self.bng.step(10)
        self.vehicle.sensors.poll()

    def step(self, steering: float, throttle: float, brake: float):
        self.vehicle.control(steering=steering, throttle=throttle, brake=brake, parkingbrake=0.0)
        self.bng.step(1)
        self.vehicle.sensors.poll()

    def close(self):
        self.bng.close()

    @property
    def pos(self): return np.array(self.vehicle.state['pos'], dtype=np.float32)
    @property
    def vel(self): return np.array(self.vehicle.state['vel'], dtype=np.float32)
    @property
    def direction(self): return np.array(self.vehicle.state['dir'], dtype=np.float32)
    @property
    def up(self): return np.array(self.vehicle.state['up'], dtype=np.float32)
    @property
    def speed(self): return float(np.linalg.norm(self.vel))


# ─── 任务逻辑层 ─────────────────────────────────────────────────────────
class OffroadDrivingTask:
    def __init__(self, task='cruise', waypoint_file=None, **kwargs):
        self._task = task
        self._load_waypoints(waypoint_file)
        self.reset_task()

    def _load_waypoints(self, file_path):
        target_path = file_path
        with open(target_path, 'r') as f:
                wp_data = json.load(f)
                self.waypoints = [np.array([wp['x'], wp['y'], wp['z']], dtype=np.float32) for wp in wp_data['waypoints']]
                self.num_waypoints = len(self.waypoints)

    def reset_task(self):
        self.current_wp_index = 1
        self.current_step_count = 0
        self.stuck_steps = 0
        self._info = {}

    def get_state_observation(self, physics: BeamNGPhysics) -> dict:
        self.target_pos = self.waypoints[min(self.current_wp_index, self.num_waypoints - 1)] # 目标点位置
        self.car_pos = physics.pos # 车辆当前位置
        self.car_forward = physics.direction # 车辆朝向向量
        self.car_up = physics.up 
        self.car_vel = physics.vel # 车辆当前速度

        state_vector = np.concatenate([
            self.target_pos,    # 3
            self.car_pos,       # 3
            self.car_forward,   # 3
            self.car_up,        # 3
            self.car_vel,       # 3
        ]).astype(np.float32)
        
        return {
            'state': state_vector
        }

    def get_reward_and_terminated(self, physics):
        reward = 0.0
        done = False
        info = {
            'success': False,     
            'terminated': False   
        }

        speed_threshold = 0.5   # 速度阈值：低于 0.5 m/s 视为停滞
        max_stuck_steps = 50    # 容忍的最大连续停滞步数（根据你的步长调整，50步约几秒钟）

        if physics.speed < speed_threshold:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0  # 只要车动起来了，计数器立刻清零

        # 如果连续停滞超过上限，直接结束回合
        if self.stuck_steps >= max_stuck_steps:
            reward -= 1.0        # 给予停滞惩罚，防止 AI 学会“消极怠工”
            done = True
            info['terminated'] = True
            info['success'] = False
            return float(reward), done, info

        current_dist = np.linalg.norm(self.target_pos - self.car_pos)

        if hasattr(self, 'last_dist'):
            dist_reduction = self.last_dist - current_dist
            reward += np.tanh(dist_reduction / 2) * 1.0

        if self.car_up[2] < 0.0:
            reward -= 1.0
            done = True
            info['terminated'] = True
            return float(reward), done, info

        if current_dist < 3:
            reward += 1.0
            self.current_wp_index += 1
            
            if self.current_wp_index < self.num_waypoints:
                self.target_pos = self.waypoints[self.current_wp_index]
                self.last_dist = np.linalg.norm(self.target_pos - self.car_pos)
            else:
                done = True
                info['terminated'] = True
                info['success'] = True
        else:
            self.last_dist = current_dist

        return float(reward), done, info

# ─── 环境封装层 ─────────────────────────────────────────────────────────
class BeamNGEnvironment(gym.Env):
    def __init__(self, physics: BeamNGPhysics, task: OffroadDrivingTask, **kwargs):
        self.physics = physics
        self.task = task
        self.step_count = 0

        # 动作空间：2维 (转向, 纵向)，范围 [-1, 1]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 观测空间
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

    def seed_act(self) -> np.ndarray:
        if not hasattr(self, '_last_seed_action'):
            self._last_seed_action = np.zeros(2, dtype=np.float32)

        target_action = np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)
        
        if np.random.rand() < 0.8:
            target_action[1] = np.random.uniform(0.0, 1.0) 
        
        
        if self.step_count < 100:
            alpha = 0.1
        else:
            alpha = 1.0
        action = (1.0 - alpha) * self._last_seed_action + alpha * target_action
        
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        self._last_seed_action = action
        if self.step_count < 100:
            self.step_count += 1
        else:
            self.step_count = 0
        print(f'self.step_count: {self.step_count}')
        
        return action
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        if np.isnan(action).any():
            print("🚨 警告: 神经网络输出了 NaN! 动作已强制清零。请检查奖励是否过大。")
            action = np.zeros_like(action)
        
        action = np.clip(action.flatten(), -1.0, 1.0)
        steering = float(action[0])
        longitudinal = float(action[1])

        if longitudinal > 0:
            throttle = longitudinal
            brake = 0.0
        else:
            throttle = 0.0
            brake = abs(longitudinal)

        print(f'转向角: {steering}')
        print(f'油门: {throttle}')
        print(f'刹车: {brake}')
        self.physics.step(steering=steering, throttle=throttle, brake=brake)
        state_obs = self.task.get_state_observation(self.physics)
        state_obs = self.state_obs_to_array(state_obs)
        reward, done, info = self.task.get_reward_and_terminated(self.physics)
        print(f'奖励: {reward}')

        return state_obs, reward, done, info
    
    def reset(self) -> np.ndarray:
        self.physics.reset()
        self.task.reset_task()
        self.step_count = 0
        state_obs = self.task.get_state_observation(self.physics)
        state_obs = self.state_obs_to_array(state_obs)
        return state_obs
    
    def state_obs_to_array(self, state_obs):
        return torch.from_numpy(
            np.concatenate([v.flatten() for v in state_obs.values()], dtype=np.float32))

    def render(self, camera_id=None) -> np.ndarray:
        cam_data = self.physics.camera.poll() 
        raw_img = cam_data['colour']
        img_arr = np.array(raw_img, dtype=np.uint8)
        
        if img_arr.shape[-1] == 4:
            img_rgb = img_arr[..., :3]
        else:
            img_rgb = img_arr
            
        return img_rgb
    
    def close(self):
        self.physics.close()