# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.go2.env_cfg import Go2EnvCfg, FOOT_LINK_NAME, BASE_LINK_NAME, JOINT_NAMES
from robot_lab.tasks.go2.mdp.terrains import (
    Go2TerrainGeneratorCfg,
    WaveTerrainCfg,
    with_slope_threshold,
)


# ==============================================================================
# 方案 A：温和混合起伏地形生成器 (30% 平地 + 40% 缓步台阶 + 30% 缓坡/波浪)
# 物理强制抬膝核心机制：
# 阶梯高度精确限制在 3~8cm 黄金区间：
#   * 若策略贴地拖行（抬腿仅 1~2cm），足端与小腿必猛烈撞击 3~8cm 台阶边缘，触发碰撞重罚；
#   * 物理障碍直接封死“贴地滑冰”的可行性，倒逼策略在数百轮内自发激发出 8~12cm 的主动大折膝；
#   * 且台阶坡度温和，不破坏整机的连续 Trot 节奏，兼顾平地超稳巡航。
# ==============================================================================
MILD_TERRAIN_CFG = Go2TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    sub_terrain_border_width=0.5,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_gym_difficulty=False,
    use_cache=False,
    sub_terrains={
        # 1. 30% 纯平网格面：保持平地优雅稳定 trot 步态基准
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
        
        # 2. 20% 缓上台阶：踏步高度 3~8cm，彻底打破贴地拖步
        "stairs_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidStairsTerrainCfg(
                proportion=0.20,
                step_height_range=(0.03, 0.08),
                step_width=0.31,
                platform_width=3.0,
            ),
            0.25,
        ),
        
        # 3. 20% 缓下台阶：下探高度 3~8cm，强化四足着地支撑平衡与送腿
        "stairs_down": with_slope_threshold(
            terrain_gen.HfPyramidStairsTerrainCfg(
                proportion=0.20,
                step_height_range=(0.03, 0.08),
                step_width=0.31,
                platform_width=3.0,
            ),
            0.25,
        ),
        
        # 4. 15% 缓斜坡：5%~20% 坡度（约 3°~11°），顺滑过渡
        "slope_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.15,
                slope_range=(0.05, 0.20),
                platform_width=3.0,
            ),
            10.0,
        ),
        
        # 5. 15% 缓波浪起伏：波幅 3~6cm，激活四足踝部灵巧随动与膝关节折叠
        "wave": with_slope_threshold(
            WaveTerrainCfg(
                proportion=0.15,
                amplitude_range=(0.03, 0.06),
                num_waves=4,
            ),
            10.0,
        ),
    },
)


@configclass
class Go2MildTerrainEnvCfg(Go2EnvCfg):
    """Configuration for official Scheme A mild terrain knee-lift fine-tuning.

    核心物理洗步逻辑：
    1. 切换至温和混合地形：物理倒逼四足折膝 8~12cm，彻底根除贴地拖行（拖步即撞台阶）；
    2. 下调标称站立高度至 0.33m，高度惩罚设为 -2.0：释放充裕膝关节折叠提腿空间；
    3. 官方标准直行对称惩罚 x_command_hip_regular = -0.5：杜绝左右髋偏差，航向笔直；
    4. 强化非预期碰撞惩罚 undesired_contacts (-1.5)：一旦腿骨蹭台阶立刻重罚；
    5. 强化 CTS 抬腿正则 feet_regulation (-0.06)：足端贴地高速送腿时强力惩罚；
    6. 弃用有数学作弊漏洞的 feet_air_time（避免网络通过 100% 贴地钻空子避罚）。
    """

    def __post_init__(self):
        super().__post_init__()

        # 1. 硬件显存安全设定 (RTX 4060 8GB 显存稳定 ~6.9GB，绝对防 OOM)
        self.scene.num_envs = 2048

        # 2. 地形完全设定为温和混合地形生成器
        self.scene.terrain.terrain_generator = MILD_TERRAIN_CFG
        self.curriculum.terrain_levels = None

        # 3. 速度指令范围设置 (全面对齐官方成熟标准并实施原地自转专项注入与y向提速)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)  # y提速至官方成熟安全红线 ±1.0 m/s
        self.commands.base_velocity.ranges.ang_vel_yaw = (-1.5, 1.5)
        self.commands.base_velocity.command_range_curriculum = []  # 立即全域生效，不再受课程压制
        
        # 启用官方边界采样 (Boundary Sampling，概率 0.2)，加入 0 允许纯侧移与纯原地自转边界
        self.commands.base_velocity.limit_vel_prob = 0.2
        self.commands.base_velocity.limit_vel = {
            "lin_vel_x": [-1, 0, 1],
            "lin_vel_y": [-1, 0, 1],
            "ang_vel_yaw": [-1, 0, 1],
        }
        self.commands.base_velocity.limit_vel_invert_when_continuous = True

        # 原地自转温和推进 (严格遵循官方纯净机制：15% 静止环境中的 35% 注入自转，占比 ~5.25%)
        self.commands.base_velocity.zero_command_curriculum = {
            "start_iter": 0,
            "end_iter": 1,
            "start_value": 0.15,
            "end_value": 0.15,
        }
        self.commands.base_velocity.limit_ang_vel_at_zero_command_prob = 0.35

        # 4. 出生稳态：贴地平稳出生，防止随机高空下砸
        self.events.reset_base.params["pose_range"]["z"] = (0.02, 0.06)
        self.events.reset_base.params["velocity_range"]["z"] = (0.0, 0.0)

        # 5. 核心奖励配比 (对齐官方多地形标准与 Commit 28b4516)
        # 5.0 对齐官方 Commit 28b4516 关键速度跟踪权重 (强化角速度爆发与侧向推力)
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.65  # 放宽核宽消除梯度饱和，强力拉起高速自转梯度

        # 5.1 标称站高调回 0.33m，高度惩罚设为 -2.0（允许生物 SLIP 弹性起伏）
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.curriculum.base_height_l2 = None
        self.curriculum.base_linear_velocity = None

        # 5.2 官方直行左右髋对称惩罚 (|FL_hip+FR_hip| + |RL_hip+RR_hip|) * x_ratio = -0.5 (已解耦动态门控)
        self.rewards.x_command_hip_regular.weight = -0.5

        # 5.3 彻底移除大腿/小腿 action_mirror（大腿相位在 trot 中天然相反，禁止对称约束）
        if hasattr(self.rewards, "action_mirror"):
            delattr(self.rewards, "action_mirror")

        # 5.4 彻底移除有数学作弊漏洞的 feet_air_time（由物理台阶直接倒逼抬膝，不给网络贴地免罚的机会）
        if hasattr(self.rewards, "feet_air_time"):
            delattr(self.rewards, "feet_air_time")

        # 5.5 强化小腿/大腿非预期碰撞惩罚 (-1.5)：一旦腿骨蹭台阶立刻重罚，强迫将足端提到台阶之上
        self.rewards.undesired_contacts.weight = -1.5

        # 5.6 官方纯净物理阻尼：微调 feet_regulation 贴地打滑惩罚至 -0.08 (严惩原地扭腰打滑摩擦，自然倒逼轻盈离地小踏步)
        self.rewards.feet_regulation.params["base_height_target"] = 0.33
        self.rewards.feet_regulation.weight = -0.08

        # 5.7 动作平滑与加速度阻尼对齐官方标准
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -1.0e-7

        # 5.8 关节姿态正则对齐官方
        self.rewards.joint_pos_penalty_l1.weight = -0.01
        self.rewards.hip_pos_penalty_l1.weight = -0.05

