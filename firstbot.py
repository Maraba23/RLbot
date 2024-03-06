import rlgym
from stable_baselines3 import PPO
from rlgym.utils.reward_functions.common_rewards import LiuDistanceBallToGoalReward, VelocityBallToGoalReward, BallYCoordinateReward
from rlgym.utils.reward_functions.common_rewards import RewardIfClosestToBall, RewardIfTouchedLast, RewardIfBehindBall
from rlgym.utils.reward_functions.common_rewards import VelocityReward, SaveBoostReward, ConstantReward, AlignBallGoal
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, LiuDistancePlayerToBallReward, FaceBallReward, TouchBallReward
# combine rewards
from rlgym.utils.reward_functions.combined_reward import CombinedReward

# combine rewards
reward_fn = CombinedReward.from_zipped(
    LiuDistanceBallToGoalReward(own_goal=False),
    VelocityBallToGoalReward(own_goal=False, use_scalar_projection=False),
    BallYCoordinateReward(exponent=1),
    VelocityReward(negative=False),
    SaveBoostReward(),
    ConstantReward(),
    AlignBallGoal(defense=1., offense=1.),
    LiuDistancePlayerToBallReward(),
    VelocityPlayerToBallReward(use_scalar_projection=False),
    FaceBallReward(),
    TouchBallReward(aerial_weight=0.)
)

#Make the default rlgym environment
env = rlgym.make(reward_fn=reward_fn)

#Initialize PPO from stable_baselines3
model = PPO("MlpPolicy", env=env, verbose=1)

#Train our agent!
model.learn(total_timesteps=100000)

#Save the model to a file
model.save("models/ppo_example")

