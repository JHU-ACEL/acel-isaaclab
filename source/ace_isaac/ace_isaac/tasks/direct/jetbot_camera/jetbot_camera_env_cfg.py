from ace_isaac.robots.jetbot import JETBOT_CONFIG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



@configclass
class JetbotSceneCfg(InteractiveSceneCfg):

    #room_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd"))
    
    jetbot: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = CameraCfg(
        data_types=["rgb"],
        prim_path="{ENV_REGEX_NS}/Robot/chassis/rgb_camera/jetbot_camera",
        spawn=None,
        height=224,
        width=224,
        update_period=.1
    )

    goal_marker = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/marker", 
                                 spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd"), 
                                 init_state=RigidObjectCfg.InitialStateCfg(pos=(.6,.3,0)))


@configclass
class JetbotCameraEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0

    action_scale = 100.0
    state_space = 0

    # scene
    scene: InteractiveSceneCfg = JetbotSceneCfg(num_envs=18, env_spacing=20.0, replicate_physics=True)
    dof_names = ["left_wheel_joint", "right_wheel_joint"]

    num_channels = 3
    # num_observations = num_channels * scene.camera.height * scene.camera.width

    action_space = 2
    state_space = 0
    observation_space = [scene.camera.height, scene.camera.width, 3]
    #num_channels * scene.camera.height * scene.camera.width
    # [tiled_camera.height, tiled_camera.width, 3]

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)