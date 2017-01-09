#coding=utf-8
#author=godpgf


class Agent(object):
    def __init__(self, env, policy):
        self.env = env
        #每个agent都有自己的策略,也可以几个agent共同用一个策略
        self.policy = policy
        #记录自己经历的所有状态
        self.states = []

    def reset(self):
        self.states = []

    def takeAction(self):
        raise NotImplementedError






