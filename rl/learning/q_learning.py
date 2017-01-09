#coding=utf-8
#author=godpgf

import random
from ..unit import *


class QLearningAgent(Agent):

    def __init__(self, env, policy, episode_min = None, episode_max = None):
        super.__init__(env, policy)
        self.episode_min = episode_min
        self.episode_max = episode_max

    def get_episode_num(self):
        return self.episode_min if self.episode_max is None else random.randint(self.episode_min, self.episode_max)

    def train(self, t_max):
        for t_cur in range(t_max):
            s = self.env.start()
            e_num = self.get_episode_num()
            while s is not None and (e_num is None or e_num > 0):
                action = self.policy.get_action(self.env, s, t_cur, t_max)
                reward, new_s = self.env.step(s, action)
                if new_s is not None:
                    self.policy.feedback(self.env, s, new_s, action, reward)
                s = new_s
