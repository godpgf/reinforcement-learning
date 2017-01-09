#coding=utf-8
#author=godpgf


#环境是状态的管理者,它可以包含一部分不可改变的状态
class Environment(object):

    #得到最开始的状态
    def start(self):
        raise NotImplementedError

    #环境下使用某个行为,并得到反馈和下一个状态,如果当前是终点,返回None
    def step(self, state, action):
        raise NotImplementedError

    def isEnd(self, state):
        raise NotImplementedError