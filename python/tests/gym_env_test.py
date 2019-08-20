import gym

from gym.envs.classic_control import rendering


env = gym.make('LunarLander-v2')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

env.close()