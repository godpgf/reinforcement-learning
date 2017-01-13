#coding=utf-8
#author=godpgf
import numpy as np


#策略对象负责根据当前的状态产生行为,它可以包含一部分不可改变的状态
class Policy(object):

    #得到某个状态下可以用的所有行为
    def getActions(self, state):
        raise NotImplementedError

    #根据当前状态产生一个行为,如果t_cur=None将不去试错,直接贪心
    def chooseAction(self, state, nextActions, t_cur = None, t_max = None):
        raise NotImplementedError

    #值函数或动作值函数
    def estimation(self, state, action = None):
        raise NotImplementedError

    #反馈奖励修正策略
    def feedback(self, cur_state, next_state, action, reward):
        raise NotImplementedError

    #如果不是终点,返回None,否则返回直接反馈
    def isEnd(self, state):
        raise NotImplementedError

    #返回下一个状态和直接奖励
    def step(self, state, action):
        raise NotImplementedError


    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    #蒙特卡洛发反馈
    def feedReward(self, states, reward):
        raise NotImplementedError

    #Q-Learning反馈
    def updateReward(self, states):
        raise NotImplementedError

    def feedState(self, state):
        raise NotImplementedError

    #用explore greedy算法得到行为
    def chooseExploreGreedyAction(self, state, nextActions, exploreRate):
        if np.random.binomial(1, exploreRate):
            np.random.shuffle(nextActions)
            return nextActions[0]
        values = []
        for a in nextActions:
            values.append((self.estimation(state,a),a))
        #np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        return values[0][1]