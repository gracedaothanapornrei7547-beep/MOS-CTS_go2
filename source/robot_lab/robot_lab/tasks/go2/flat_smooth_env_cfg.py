from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.go2.env_cfg import Go2EnvCfg, FOOT_LINK_NAME
from robot_lab.tasks.go2.mdp.terrains import Go2TerrainGeneratorCfg


# 100% 纯平网格地面生成器 (消除台阶、障碍、波浪等一切地形干扰，提供纯粹平地流场)
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
class Go2FlatSmoothEnvCfg(Go2EnvCfg):
    """Configuration for official-recipe knee-lift & airborne de-jittering fine-tuning (~30 mins / 1000 iters).

    官方标准配方核心物理洗步逻辑：
    1. 锁定 100% 纯平网格面：彻底隔离多地形冲击干扰，提供纯净平地流场；
    2. 下调标称站立高度至 0.33m（官方标称值，此前 0.38m 导致四腿紧绷无法折叠）：释放膝关节充裕折叠空间；
    3. 高度惩罚降回 -2.0（此前 -10.0 过激惩罚导致策略被迫贴地滑行）：允许 trot 腾空时机身有自然的弹性起伏；
    4. 彻底移除大腿/小腿 action_mirror（大腿相位在 trot 中天然相反，禁止对称约束）：仅保留官方成熟的 x_command_hip_regular = -0.5 约束左右髋；
    5. 强化 8cm 提膝腾空 feet_regulation (-0.08)：强制摆动相快速送腿时足端必须离地 8cm 以上，彻底根除贴地摩擦颤振。
    """

    def __post_init__(self):
        super().__post_init__()

        # 1. 硬件显存安全设定 (RTX 4060 8GB VRAM 稳定占用 6.9GB，绝对防 OOM)
        self.scene.num_envs = 2048

        # 2. 地形完全锁定为 100% 纯平网格面，清除多地形课程
        self.scene.terrain.terrain_generator = FLAT_TERRAIN_CFG
        self.curriculum.terrain_levels = None

        # 3. 速度指令范围设置 (聚焦平地巡航与高速前向 Trot 步态结晶)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_yaw = (-1.0, 1.0)

        # 4. 出生稳态：严禁随机空中下砸，贴地平稳出生
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.05)
        self.events.reset_base.params["velocity_range"]["z"] = (0.0, 0.0)

        # 5. 核心奖励配比 (严格对齐官方标准配方，彻底根治贴地拖步与虚踩)
        # 5.1 标称站高调回 0.33m，高度惩罚降回 -2.0（允许自然生物起伏，释放提腿空间）
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.curriculum.base_height_l2 = None
        self.curriculum.base_linear_velocity = None

        # 5.2 官方直行左右髋对称惩罚 (|FL_hip+FR_hip| + |RL_hip+RR_hip|) * x_ratio = -0.5
        self.rewards.x_command_hip_regular.weight = -0.5

        # 5.3 彻底移除大腿/小腿 action_mirror（释放大腿一前一后自由相位，避免网络被迫滑行）
        if hasattr(self.rewards, "action_mirror"):
            delattr(self.rewards, "action_mirror")

        # 5.4 引入官方同款空中时间强力奖励 feet_air_time (打破 360ms 滞地与滞控，强制腾空 >= 0.20s)
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
                "threshold": 0.20,
            },
        )

        # 5.5 强化 8cm 提膝腾空正则 feet_regulation (-0.06)
        self.rewards.feet_regulation.params["base_height_target"] = 0.33
        self.rewards.feet_regulation.weight = -0.06

        # 5.6 松绑动作平滑与加速度阻尼 (释放小腿 1.8~2.0 rad 大幅提膝爆发力，消除 120ms 电机滞后)
        self.rewards.action_rate_l2.weight = -0.008
        self.rewards.action_smoothness_l2.weight = -0.008
        self.rewards.joint_acc_l2.weight = -1.5e-7

        # 5.7 关节姿态正则对齐官方
        self.rewards.joint_pos_penalty_l1.weight = -0.01
        self.rewards.hip_pos_penalty_l1.weight = -0.05

