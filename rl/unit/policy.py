#coding=utf-8
#author=godpgf


#策略对象负责根据当前的环境产生行为
class Policy(object):

    #根据当前环境产生一个行为,如果t_cur=None将不去试错,直接贪心
    def getAction(self, env, state, t_cur = None, t_max = None):
        raise NotImplementedError

    #反馈奖励修正策略
    def feedback(self, env, cur_state, next_state, action, reward):
        raise NotImplementedError