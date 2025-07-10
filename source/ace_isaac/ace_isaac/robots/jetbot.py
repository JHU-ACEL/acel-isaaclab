import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

'''

# Documentation for Writing an ArticulationCfg: https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html
# Jackal joint names: ['front_left_wheel_joint', 'front_right_wheel_joint', 'rear_left_wheel_joint', 'rear_right_wheel_joint']

JACKAL_CONFIG = ArticulationCfg(
    # This imports the actual robot usd file
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/Jackal/jackal.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={       
            "front_left_wheel_joint": 0.0,
            "front_right_wheel_joint": 0.0,
            "rear_left_wheel_joint": 0.0,
            "rear_right_wheel_joint": 0.0,
        }
    ),
    # This states what kind of control each joint's motor will use. ImplicitActuatorCfg uses the default PD controller for each motor that controls the joint 
    # Documentation/Source: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],   # matches all wheel joints
            effort_limit=None,              # torque limit per wheel
            velocity_limit=None,             # rad/s per wheel
            stiffness=None,                  # use default PD gains
            damping=None,
        )
    },  
)


'''