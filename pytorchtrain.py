import gym

from stable_baselines.common.policies import MlpPolicy,RecurrentActorCriticPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

from ridicare import HumanoidStandupEnv
om=HumanoidStandupEnv()
om = make_vec_env('HumanoidStandup-v2', n_envs=4)

model = PPO2(MlpPolicy, om, verbose=1,nminibatches=4, n_steps=250, gamma=0.285)
model.learn(total_timesteps=1000000)
model.save("ppo6")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo6")

obs = om.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = om.step(action)
    om.render()