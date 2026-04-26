import hydra
from beamngpy import BeamNGpy, Scenario, Vehicle

@hydra.main(config_name='config', config_path='.')
def testBeamNGphysics(cfg: dict):
	BeamNGPhysics(cfg)

import json
from beamngpy.sensors import Electrics
# from beamngpy.sensors import Damage
# from beamngpy.sensors import Camera
# from beamngpy.sensors import Radar
# from beamngpy.sensors import AdvancedIMU
class BeamNGPhysics:
    def __init__(self, cfg):
        self.cfg = cfg

        # 启动仿真器
        self.bng = BeamNGpy(host=cfg.beamng_host, port=cfg.beamng_port, home=cfg.beamng_home)
        self.bng.open(launch=True)
        
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
        self._add_sensors()
        self._add_npcs()
        self.scenario.make(self.bng)

        # 加载、启动场景
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()

        # npc 巡逻逻辑
        for npc in self.npcs:
            npc.ai.set_mode('traffic') # 自主巡逻
        
        # 运行 60 帧以调整初始状态
        self.bng.step(60)

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
        # self.imu = AdvancedIMU('ego_imu', self.bng, self.vehicle, 
        #     pos=(0, 0, 1.0),      # 安装在车辆质心附近
        #     dir=(0, -1, 0),       # 朝向正前方
        #     up=(0, 0, 1)          # Z轴朝上
        # )

        # [RGB-D 相机]：输出彩色图像和深度图
        # self.camera = Camera('front_cam', self.bng, self.vehicle,
        #     pos=(0, -0.5, 1.5),     # 安装在挡风玻璃或车顶上方
        #     dir=(0, -1, 0),         # 朝向车头正前方
        #     field_of_view_y=70,     # FOV 视野大小
        #     resolution=(256, 256),  # 分辨率（RL 推荐降采样以加快训练）
        #     is_render_colors=True,  # 开启彩色渲染
        #     is_render_depth=True    # 开启深度图渲染
        # )

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
            scales=[(5,5,5)] * len(self.waypoints),
            ids=[f'wp_{i}' for i in range(len(self.waypoints))]
        )
    
    def _add_npcs(self):
        self.npcs = []
        start_x, start_y, start_z = self.vehicle_start_pos # 放在 ego 起始点旁边 10 米处测试
        
        npc_vehicle_positions = [
            (start_x + 10, start_y, start_z + 0.5),
            (start_x - 10, start_y, start_z + 0.5)
        ]
        
        for i, pos in enumerate(npc_vehicle_positions):
            npc = Vehicle(f'npc_{i}', model='pickup', licence='TEST')
            self.scenario.add_vehicle(npc, pos=pos, rot_quat=(0, 0, 0, 1))
            self.npcs.append(npc)
    
if __name__ == '__main__':
	testBeamNGphysics()