import rlgym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, LiuDistanceBallToGoalReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

# Função para criar o ambiente RLGYM com configurações específicas para self-play
def make_env():
    reward_function = CombinedReward(reward_functions=[VelocityBallToGoalReward(), LiuDistanceBallToGoalReward()])  # Exemplo de recompensa combinada
    terminal_conditions = [TimeoutCondition(180)]  # Exemplo de condição terminal: tempo máximo de jogo de 180 segundos

    return rlgym.make(
        reward_fn=reward_function,
        terminal_conditions=terminal_conditions,
        obs_builder=AdvancedObs(),
        self_play=True  # Habilita self-play
    )

# Envolve o ambiente RLGYM para uso com Stable Baselines 3
env = SB3MultipleInstanceEnv(make_env, num_instances=1)  # num_instances define quantas cópias do ambiente serão criadas para treinamento paralelo

# Configuração e treinamento do modelo
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)
