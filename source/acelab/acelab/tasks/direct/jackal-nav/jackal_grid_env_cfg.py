# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from acelab.robots.jackal.jackal import JACKAL_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg


@configclass
class JackalGridEnvCfg(DirectRLEnvCfg):

    episode_length_s = 100.0

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


    # - spaces definition
    state_space = 0
    action_space = 4
    observation_space = [5, tiled_camera.height, tiled_camera.width, 3]
    #observation_space = [tiled_camera.height, tiled_camera.width, 3]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=400, env_spacing=50.0, replicate_physics=True)

    dof_names = ['front_left_wheel_joint', 'front_right_wheel_joint', 'rear_left_wheel_joint', 'rear_right_wheel_joint']