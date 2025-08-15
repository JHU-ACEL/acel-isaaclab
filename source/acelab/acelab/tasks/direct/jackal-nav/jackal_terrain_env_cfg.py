# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from acelab.robots.jackal.jackal import JACKAL_CONFIG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg

import isaaclab.sim as sim_utils


@configclass
class MarsTerrainSceneCfg(InteractiveSceneCfg):
    """
    Mars Terrain Scene Configuration
    """
    # Hidden Terrain (merged terrain of ground and obstacles) for raycaster.
    # This is done because the raycaster doesn't work with multiple meshes
    # hidden_terrain = AssetBaseCfg(
    #     prim_path="/World/terrain/hidden_terrain",
    #     spawn=sim_utils.UsdFileCfg(
    #         visible=False,
    #         usd_path=os.path.join(
    #             os.path.dirname(os.path.abspath(__file__)),
    #             "terrain1",
    #             "terrain_merged.usd",
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    # )

    # Obstacles
    obstacles = AssetBaseCfg(
        prim_path="/World/terrain/obstacles",
        spawn=sim_utils.UsdFileCfg(
            visible=True,
            usd_path= "/home/bchien1/ACE_IsaacLabInfrastructure/source/acelab/acelab/terrain/mars_terrain/rocks_merged.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Ground Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="usd",
        usd_path="/home/bchien1/ACE_IsaacLabInfrastructure/source/acelab/acelab/terrain/mars_terrain/terrain_only.usd",
    )

@configclass
class JackalTerrainEnvCfg(DirectRLEnvCfg):

    episode_length_s = 5.0

    # simulation
    decimation = 2
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = JACKAL_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    
    # sensors
    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/base_link/bumblebee_stereo_camera_frame/bumblebee_stereo_right_frame/bumblebee_stereo_right_camera",
    #     #offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    #     data_types=["rgb"],
    #     spawn=None,
    #     width=64,
    #     height=64,
    # )

    # - spaces definition
    state_space = 0
    action_space = 4
    observation_space = 3
    #observation_space = [tiled_camera.height, tiled_camera.width, 3]

    # scene
    scene: MarsTerrainSceneCfg = MarsTerrainSceneCfg(num_envs=1, env_spacing=0.0, replicate_physics=True)

    dof_names = ['front_left_wheel_joint', 'front_right_wheel_joint', 'rear_left_wheel_joint', 'rear_right_wheel_joint']