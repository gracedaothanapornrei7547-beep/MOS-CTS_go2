# Copyright (c) 2024-2026
# Registration for ObstacleDog Gym tasks

import gymnasium as gym

gym.register(
    id="RobotLab-ObstacleDog-Flat-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ObstacleDogFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:ObstacleDogPPORunnerCfg",
    },
)
