#coding=utf-8
#author=godpgf


class Agent(object):
    def __init__(self, policy):
        #每个agent都有自己的策略,也可以几个agent共同用一个策略
        self.policy = policy
        #记录自己经历的所有状态或者动作状态
        self.states = []

    def reset(self):
        self.states = []

    # accept a state
    def feedState(self, state):
        self.states.append(state)
        self.policy.feedState(state)

    def isEnd(self):
        return self.policy.isEnd(self.states[-1])

    def takeAction(self, t_cur = None, t_max = None):
        return self.policy.chooseAction(self.states[-1], self.policy.getActions(self.states[-1]))

    def step(self, action):
        return self.policy.step(self.states[-1], action)

    def savePolicy(self):
        self.policy.save()

    def loadPolicy(self):
        self.policy.load()

    def feedReward(self, reward):
        self.policy.feedReward(self.states, reward)

    def updateReward(self):
        self.policy.updateReward(self.states)






