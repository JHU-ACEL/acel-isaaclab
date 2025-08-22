# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from acelab.robots.jackal.jackal import JACKAL_CONFIG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class JackalGridEnvCfg(DirectRLEnvCfg):

    # Change to 20s ~ 30s when training on the Easy curriculum
    episode_length_s = 25.0

    # simulation
    decimation = 2
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = JACKAL_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    
    # sensors
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/bumblebee_stereo_camera_frame/bumblebee_stereo_right_frame/bumblebee_stereo_right_camera",
        #offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=None,
        width=64,
        height=64,
    )

    # obs_cfg = RigidObjectCfg(prim_path="/World/envs/env_.*/marker", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd", scale = (10.0, 100.0, 100.0)), 
    #                          init_state = RigidObjectCfg.InitialStateCfg(pos=(1.5,0,0.05)))
    

    # lidar: RayCasterCfg = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base_link/visuals/mesh_6",
    #     update_period = 0.1,
    #     max_distance = 10.0,
    #     #offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.5)),
    #     pattern_cfg=patterns.GridPatternCfg(
    #         resolution=1.0,
    #         size=(1.0, 0.0),
    #         direction=(1.0, 0.0, 0.0),
    #     ),
    #     debug_vis=True,
    #     ray_alignment = "yaw",
    #     mesh_prim_paths = ["/World/envs/env_0/marker"],
    #     #mesh_prim_paths = ["/World/ground"]
    # )
    
    # - spaces definition
    state_space = 0
    action_space = 4
    observation_space = [5, tiled_camera.height, tiled_camera.width, 3]
    #observation_space = [tiled_camera.height, tiled_camera.width, 3]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=500, env_spacing=50.0, replicate_physics=True) # Change num_envs to 500 when training

    dof_names = ['front_left_wheel_joint', 'front_right_wheel_joint', 'rear_left_wheel_joint', 'rear_right_wheel_joint']