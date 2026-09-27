"""Environment factory for classic control and MuJoCo tasks."""
import gymnasium as gym


CLASSIC = {
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
}
MUJOCO = {
    "Hopper-v4", "Walker2d-v4", "HalfCheetah-v4",
    "Ant-v4", "InvertedPendulum-v4",
}


def make_env(name):
    if name in CLASSIC or name in MUJOCO:
        return gym.make(name)
    raise ValueError(f"Unknown environment: {name}")