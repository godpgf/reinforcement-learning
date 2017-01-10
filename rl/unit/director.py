#coding=utf-8
#author=godpgf


class Director(object):

    #训练一个策略,训练次数是t_max
    def train(self, t_max):
        raise NotImplementedError

    def play(self, t_cur, t_max):
        raise NotImplementedError