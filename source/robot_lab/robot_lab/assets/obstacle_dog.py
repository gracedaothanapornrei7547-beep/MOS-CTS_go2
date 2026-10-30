# Copyright (c) 2024-2026
# Custom Obstacle Dog Articulation Configuration for Isaac Lab

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

##
# Configuration
##

OBSTACLE_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/rum/obstacle_dog/usd/obstacle_dog.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            "joint.*_1": 0.0,   # Hip roll
            "joint.*_2": 0.6,   # Thigh pitch
            "joint.*_3": -1.2,  # Calf pitch
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=["joint.*"],
            effort_limit=34.0,
            saturation_effort=34.0,
            velocity_limit=25.0,
            stiffness=35.0,
            damping=0.8,
            friction=0.02,
        ),
    },
)
