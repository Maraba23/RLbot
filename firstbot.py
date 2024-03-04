import rlgym
from stable_baselines3 import PPO
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward, VelocityBallToGoalReward, BallYCoordinateReward

# Create the environment
env = rlgym.make("default", self_play=True, spawn_opponents=False, team_size=1, tick_skip=8, reward_fn=[LiuDistancePlayerToBallReward(), VelocityBallToGoalReward(), BallYCoordinateReward()])

# Train the agent
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100000)

# Save the agent
model.save("models/ppo_default")