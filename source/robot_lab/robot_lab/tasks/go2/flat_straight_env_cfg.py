from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.go2.env_cfg import Go2EnvCfg
from robot_lab.tasks.go2.mdp.terrains import Go2TerrainGeneratorCfg


# 100% 纯平网格地面生成器 (保留完整张量接口，消除障碍碰撞开销)
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
class Go2FlatStraightEnvCfg(Go2EnvCfg):
    """Configuration for training Go2 on flat ground with 100% official pure rewards."""

    def __post_init__(self):
        super().__post_init__()

        # 1. 硬件显存适配 (RTX 4060 Laptop 8GB VRAM) -> 4096 环境极速且防 OOM
        self.scene.num_envs = 4096

        # 2. 地形完全锁定为 100% 纯平网格面，清除地形难度课程
        self.scene.terrain.terrain_generator = FLAT_TERRAIN_CFG
        self.curriculum.terrain_levels = None

        # 3. 全向指令自然诱导 (保留官方全向速度探索空间，迫使 MoE-CTS 自然淘汰 Bound 步态)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_yaw = (-1.0, 1.0)

        # 4. 彻底锁定出生位姿：严禁开局随机偏航旋转与高空下砸 (出生贴地稳态)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.0)

        # 5. 奖励系统：100% 官方纯净版项，锁定机身站立高度惩罚权重，铲除低趴塌陷避难所
        # (继承自 Go2EnvCfg.rewards 官方 15 项标准 MDP 奖励，无任何自创项)
        self.rewards.base_height_l2.weight = -5.0
        self.curriculum.base_height_l2 = None
        self.curriculum.base_linear_velocity = None

        # 6. 髋关节侧摆收紧 (100% 官方项微调，彻底矫正后腿劈叉外展，保持双腿矢状面平行直立)
        self.rewards.hip_pos_penalty_l1.weight = -0.6

        # 7. 左右动作镜像对称奖励 (官方 MDP 奖励，消除跛行单腿拐杖/对称破缺，促成健康对角小跑)
        self.rewards.action_mirror = RewTerm(
            func=mdp.action_mirror,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mirror_joints": [
                    ["FL_hip_joint", "FR_hip_joint"],
                    ["FL_thigh_joint", "FR_thigh_joint"],
                    ["FL_calf_joint", "FR_calf_joint"],
                    ["RL_hip_joint", "RR_hip_joint"],
                    ["RL_thigh_joint", "RR_thigh_joint"],
                    ["RL_calf_joint", "RR_calf_joint"],
                ],
            },
        )

