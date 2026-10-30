from __future__ import annotations

from isaaclab.utils import configclass
from robot_lab.tasks.go2.env_cfg import Go2EnvCfg


@configclass
class Go2SmoothEnvCfg(Go2EnvCfg):
    """Configuration for 1000-iteration symmetry & smoothness fine-tuning (aligned with official go2_rl_gym).

    核心消抖与对称性调优配比：
    1. 移植官方左右髋关节对称惩罚：x_command_hip_regular 赋予 -0.5，前向奔跑时强力约束左右髋对称镜像，消灭对角线扭晃；
    2. 坚决恢复官方标称站立高度惩罚：base_height_l2 设为 -10.0，锁定机身于 0.33~0.34m 高位倒立摆平滑滑行，消除塌腰下砸；
    3. 动作变化率与平滑度：action_rate 设为 -0.02，action_smoothness 设为 -0.02，滤除指令毛刺同时避免僵死；
    4. 足端抬高正则：feet_regulation 设为 -0.08，防止贴地拖脚绊步，保障腾空足端自然展开；
    5. 锁定课程调度：固定当前成熟的 lin_vel_z 自由度与 -10.0 高度约束，平稳收敛。
    """

    def __post_init__(self):
        super().__post_init__()

        # 1. 显存安全设定 (RTX 4060 8GB 稳定占用 6.9GB，绝对防 OOM)
        self.scene.num_envs = 2048

        # 2. 核心对称性消抖奖励配比 (对齐官方 go2_rl_gym)
        # 2.1 官方核心：直行左右髋对称惩罚 (|FL+FR| + |RL+RR|) * x_ratio
        self.rewards.x_command_hip_regular.weight = -0.5

        # 2.2 恢复官方标称强站立高度惩罚 (-10.0 锁定高位滑行，消除塌腰下砸)
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.lin_vel_z_l2.weight = 0.0

        # 2.3 动作平滑与加速度抑制 (温和阻尼滤除高频电机抖动)
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.action_smoothness_l2.weight = -0.02
        self.rewards.joint_acc_l2.weight = -1.5e-7

        # 2.4 足端与名义姿态约束
        self.rewards.feet_regulation.weight = -0.08
        self.rewards.joint_pos_penalty_l1.weight = -0.02
        self.rewards.hip_pos_penalty_l1.weight = -0.05

        # 3. 课程调度锁定 (平顺微调阶段锁定权重，避免二次冲击)
        self.curriculum.base_height_l2 = None
        self.curriculum.base_linear_velocity = None
