#coding=utf-8
#author=godpgf
from rl.unit import *
import numpy as np
import pickle
import random
from TicTacToe import T3State, T3Director, HummanAgent, BOARD_ROWS, BOARD_COLS
import tensorflow as tf
from collections import deque

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev = 0.01, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name):
    initial = tf.constant(0.01, shape = shape, name=name)
    return tf.Variable(initial)


class DeepT3State(T3State):

    def getHash(self):
        return self.data.reshape(BOARD_ROWS * BOARD_COLS)

class DeepT3Policy(Policy):
    def __init__(self, symbol, stepSize = 0.16, exploreRate = 0.16):
        self.symbol = symbol
        self.stepSize = stepSize
        self.exploreRate = exploreRate
        self.x, self.readout = self.createNetwork()
        self.y = tf.placeholder("float", [None])
        self.cost = tf.reduce_mean(tf.square(self.y - self.readout))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        self.D = deque()

    def createNetwork(self):
        w1 = weight_variable([9,4],"w1_%d"%self.symbol)
        b1 = bias_variable([4],"b1_%d"%self.symbol)
        w2 = weight_variable([4,1],"w2_%d"%self.symbol)
        b2 = bias_variable([1],"b2_%d"%self.symbol)
        x = tf.placeholder("float", [None, 9])
        h = tf.nn.relu(tf.matmul(x, w1) + b1)
        readout = tf.nn.relu(tf.matmul(h,w2) + b2)
        return x, readout

    def chooseAction(self, state, nextActions, t_cur=None, t_max=None):
        if nextActions is None:
            return None
        return self.chooseExploreGreedyAction(state, nextActions, self.exploreRate)

    def getActions(self, state):
        if self.isEnd(state):
            return None
        nextPositions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    nextPositions.append([i, j])
        return nextPositions

    def step(self, state, action):
        newState = DeepT3State()
        newState.data = np.copy(state.data)
        newState.data[action[0],action[1]] = self.symbol
        reward = self.isEnd(newState)
        if reward is None:
            reward = 0
        return newState, reward

    def estimation(self, state, action = None):
        nextState, reward = self.step(state, action)
        id = nextState.getHash()
        e_reward = self.isEnd(nextState)
        if e_reward is not None:
            return e_reward
        R = self.readout.eval(feed_dict={self.x: [nextState.getHash()]})[0]
        return R[0]

    def updateReward(self, states):
        if len(states) <= 2:
            return

        endReward = self.isEnd(states[-1])
        if endReward:
            self.D.append((states[-2],states[-2],1-endReward))
        else:
            maxRewardAction = self.chooseExploreGreedyAction(states[-1], self.getActions(states[-1]), 0)
            nextState, reward = self.step(states[-1], maxRewardAction)
            #当前状态、下一个状态、奖励
            self.D.append((states[-2],nextState,reward))
        if len(self.D) > REPLAY_MEMORY:
            self.D.popleft()
            minibatch = random.sample(self.D, BATCH)
            s_batch = [d[0].getHash() for d in minibatch]
            s_batch_next = [d[1].getHash() for d in minibatch]
            readout_y_batch = self.readout.eval(feed_dict={self.x: s_batch_next})
            y_batch = []
            for i in range(len(minibatch)):
                if minibatch[i][1] is minibatch[i][0]:
                    y_batch.append(minibatch[i][2])
                else:
                    y_batch.append(minibatch[i][2] + self.stepSize * readout_y_batch[i][0])

            self.train_step.run(feed_dict = {
                self.y : y_batch,
                self.x : s_batch}
            )

    def feedState(self, state):
        pass

    def isEnd(self, state):
        if state.end is not None:
            return state.end
        results = []
        # check row
        for i in range(0, BOARD_ROWS):
            results.append(np.sum(state.data[i, :]))
        # check columns
        for i in range(0, BOARD_COLS):
            results.append(np.sum(state.data[:, i]))

        # check diagonals
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += state.data[i, i]
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += state.data[i, BOARD_ROWS - 1 - i]

        for result in results:
            if result == 3:
                state.end = 1.0 if self.symbol == 1 else 0
                return state.end
            if result == -3:
                state.end = 1.0 if self.symbol == -1 else 0
                return state.end

        sum = np.sum(np.abs(state.data))
        if sum == BOARD_ROWS * BOARD_COLS:
            state.end = 0.5
            return state.end

        state.end = None
        return state.end

    def save(self):
        pass

    def load(self):
        pass

show = False
REPLAY_MEMORY = 512 # number of previous transitions to remember
BATCH = 32 # size of minibatch


class DeepT3Director(T3Director):

    def __init__(self, sess):
        super(DeepT3Director, self).__init__()
        self.sess = sess

    def reset(self):
        self.currentState = DeepT3State()
        self.player1.reset()
        self.player2.reset()
        self.currentPlayer = None
        self.feedCurrentState()


    def train(self, epochs):
        self.player1 = Agent(DeepT3Policy(1, 0.01, 0.8))
        self.player2 = Agent(DeepT3Policy(-1))
        sess.run(tf.initialize_all_variables())
        p1Win = 0
        p2Win = 0
        for i in range(epochs):
            winner = self.play(i,epochs)
            if winner == self.player1:
                p1Win += 1
            if winner == self.player2:
                p2Win += 1
            self.reset()
        print("player1 win rate %.2f%%"%(p1Win*100.0/epochs))
        print("player2 win rate %.2f%%"%(p2Win*100.0/epochs))

        #self.player1.savePolicy()
        #self.player2.savePolicy()
        tf.train.Saver().save(self.sess,"model.ckbt")

    def compete(self, turns):
        self.player1 = Agent(DeepT3Policy(1, 0.1, 0))
        self.player2 = Agent(DeepT3Policy(-1, 0.1, 0))
        p1Win = 0
        p2Win = 0
        tf.train.Saver().restore(self.sess, "model.ckbt")
        #self.player1.loadPolicy()
        #self.player2.loadPolicy()
        for i in range(turns):
            print("Epoch", i)
            winner = self.play(i,turns)
            if winner == self.player1:
                p1Win += 1
            if winner == self.player2:
                p2Win += 1
            self.reset()
        print("player1 win rate %.2f%%"%(p1Win*100.0/turns))
        print("player2 win rate %.2f%%"%(p2Win*100.0/turns))

    def vs_human(self):
        self.player1 = HummanAgent(DeepT3Policy(1, 0.6, 0))
        self.player2 = Agent(DeepT3Policy(-1, 0.6, 0))
        self.player2.loadPolicy()
        while True:
            winner = self.play()
            if winner == self.player1:
                print("win!")
            elif winner == self.player2:
                print("lost!")
            else:
                print("tie!")
            self.reset()

sess = tf.InteractiveSession()
direct = DeepT3Director(sess)

#direct.train(20000)
#direct.train(500)
direct.vs_human()