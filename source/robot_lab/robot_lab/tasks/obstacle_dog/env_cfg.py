# Copyright (c) 2024-2026
# ObstacleDog Environment Configuration for Isaac Lab

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.assets.obstacle_dog import OBSTACLE_DOG_CFG
from robot_lab.tasks.go2.env_cfg import Go2EnvCfg
from robot_lab.tasks.go2.mdp.terrains import Go2TerrainGeneratorCfg

# 12 Actuated Joints in order
JOINT_NAMES = [
    "joint1_1", "joint1_2", "joint1_3",  # Rear-Left (RL)
    "joint2_1", "joint2_2", "joint2_3",  # Front-Left (FL)
    "joint3_1", "joint3_2", "joint3_3",  # Rear-Right (RR)
    "joint4_1", "joint4_2", "joint4_3",  # Front-Right (FR)
]

BASE_LINK_NAME = "base_link"
FOOT_LINK_NAME = "link[1-4]_3"
BASE_HEIGHT_TARGET = 0.22  # 3.36kg Mini dog nominal standing height (~22cm)

# 100% Flat terrain
FLAT_TERRAIN_CFG = Go2TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    sub_terrain_border_width=0.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_gym_difficulty=False,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


@configclass
class ObstacleDogFlatEnvCfg(Go2EnvCfg):
    """Configuration for training ObstacleDog on flat ground with PPO."""

    base_height_target = BASE_HEIGHT_TARGET

    def __post_init__(self):
        super().__post_init__()

        # 1. Robot Asset
        self.scene.robot = OBSTACLE_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 2048  # Safe memory footprint for RTX 4060 Laptop (8GB)

        # 2. Sensors binding to obstacle_dog prims
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.height_scanner_small.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"

        # 3. Action space (12 joints)
        self.actions.joint_pos.joint_names = JOINT_NAMES
        self.actions.joint_pos.scale = {"joint.*_1": 0.25, "^(?!joint.*_1).*": 0.25}

        # 4. Observations joint bindings
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = JOINT_NAMES
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = JOINT_NAMES

        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = JOINT_NAMES
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = JOINT_NAMES
        self.observations.critic.joint_acc.params["asset_cfg"].joint_names = JOINT_NAMES
        self.observations.critic.joint_torque.params["asset_cfg"].joint_names = JOINT_NAMES
        self.observations.critic.contact_force.params["sensor_cfg"].body_names = FOOT_LINK_NAME

        self.observations.single_obs.joint_pos.params["asset_cfg"].joint_names = JOINT_NAMES
        self.observations.single_obs.joint_vel.params["asset_cfg"].joint_names = JOINT_NAMES

        # 5. Domain Randomization Events
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [BASE_LINK_NAME]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = ["^(?!.*base_link).*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [BASE_LINK_NAME]

        # 6. Flat ground setup
        self.scene.terrain.terrain_generator = FLAT_TERRAIN_CFG
        self.curriculum.terrain_levels = None
        self.curriculum.base_height_l2 = None
        self.curriculum.base_linear_velocity = None

        # 7. Velocity commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_yaw = (-1.0, 1.0)

        # 8. Initial spawn pose
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)

        # 9. Termination conditions
        if hasattr(self.terminations, "illegal_contact"):
            self.terminations.illegal_contact.params["sensor_cfg"].body_names = [BASE_LINK_NAME]

        # 10. Rewards tuning for mini dog (3.36kg)
        self.rewards.base_height_l2.weight = -8.0
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [BASE_LINK_NAME]
        self.rewards.base_height_l2.params["target_height"] = BASE_HEIGHT_TARGET
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = JOINT_NAMES
        self.rewards.joint_power.params["asset_cfg"].joint_names = JOINT_NAMES
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = JOINT_NAMES
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [BASE_LINK_NAME, "link[1-4]_[12]"]
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = JOINT_NAMES
        self.rewards.feet_regulation.params["asset_cfg"].body_names = FOOT_LINK_NAME
        self.rewards.feet_regulation.params["base_height_target"] = BASE_HEIGHT_TARGET
        self.rewards.hip_pos_penalty_l1.params["asset_cfg"].joint_names = "joint.*_1"
        self.rewards.joint_pos_penalty_l1.params["asset_cfg"].joint_names = "joint.*_[23]"

        # 11. Left-Right Action Mirror symmetry
        # Leg 2 (FL) <-> Leg 4 (FR), Leg 1 (RL) <-> Leg 3 (RR)
        self.rewards.action_mirror = RewTerm(
            func=mdp.action_mirror,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mirror_joints": [
                    ["joint2_1", "joint4_1"],  # FL hip roll <-> FR hip roll
                    ["joint2_2", "joint4_2"],  # FL thigh pitch <-> FR thigh pitch
                    ["joint2_3", "joint4_3"],  # FL calf pitch <-> FR calf pitch
                    ["joint1_1", "joint3_1"],  # RL hip roll <-> RR hip roll
                    ["joint1_2", "joint3_2"],  # RL thigh pitch <-> RR thigh pitch
                    ["joint1_3", "joint3_3"],  # RL calf pitch <-> RR calf pitch
                ],
            },
        )
