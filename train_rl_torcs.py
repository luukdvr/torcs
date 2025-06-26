import gym
import gym_torcs
from stable_baselines3 import PPO

# Maak de TORCS-omgeving aan, met visuele feedback (TORCS GUI)
env = gym.make('Torcs-v0', vision=False, throttle=True, gear_change=True)

# PPO-model aanmaken of laden
try:
    model = PPO.load('ppo_torcs', env=env)
    print("Bestaand model geladen.")
except Exception:
    model = PPO('MlpPolicy', env, verbose=1)
    print("Nieuw model aangemaakt.")

# Trainen met continue reset bij crash of van de baan
TIMESTEPS = 10000
obs = env.reset()
for i in range(10):  # 10x zoveel timesteps, pas aan naar wens
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save('ppo_torcs')
    print(f"Model opgeslagen na {TIMESTEPS*(i+1)} timesteps.")
    obs = env.reset()

# Sluit de omgeving
env.close()
