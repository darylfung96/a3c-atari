import gym
import numpy as np
from scipy import misc

def _preprocess_observation(observation):
    observation = np.mean(observation, 2)
    observation = misc.imresize(observation, (47, 47)).astype(np.float32)
    observation = np.expand_dims(observation, 2)
    observation = np.expand_dims(observation, 0)
    return observation

class Env():
    def __init__(self, game_name):
        self.env = gym.make(game_name)

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        observation = _preprocess_observation(observation)
        reward = np.clip(reward, -1, 1)
        return observation, reward, done

    def render(self):
        self.env.render()

    def number_actions(self):
        return self.env.action_space.n

    def reset(self):
        observation = self.env.reset()
        return _preprocess_observation(observation)