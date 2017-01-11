#coding=utf-8
#author=godpgf
from rl.unit import *
import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class T3State(State):
    def __init__(self):
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.hashVal = None
        self.end = None

    def getHash(self):
        if self.hashVal is None:
            self.hashVal = 0
            for i in self.data.reshape(BOARD_ROWS * BOARD_COLS):
                if i == -1:
                    i = 2
                self.hashVal = self.hashVal * 3 + i
        return int(self.hashVal)

    # print the board
    def show(self):
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                if self.data[i, j] == 0:
                    token = '0'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('-------------')

class T3Policy(Policy):
    def __init__(self, symbol, stepSize = 0.1, exploreRate = 0.8):
        self.symbol = symbol
        self.stepSize = stepSize
        self.exploreRate = exploreRate
        self.estimations = dict()

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
        newState = T3State()
        newState.data = np.copy(state.data)
        newState.data[action[0],action[1]] = self.symbol
        return newState, 0

    def estimation(self, state, action = None):
        nextState, reward = self.step(state, action)
        id = nextState.getHash()
        e_reward = self.isEnd(nextState)
        if e_reward is not None:
            self.estimations[id] = e_reward
        if id in self.estimations:
            return self.estimations[id]
        else:
            return 0.5

    def feedReward(self, states, reward):
        if len(states) == 0:
            return
        states = [state.getHash() for state in states]
        target = reward
        for latestState in reversed(states):
            value = self.estimations[latestState] + self.stepSize * (target - self.estimations[latestState])
            self.estimations[latestState] = value
            target = value

    def feedState(self, state):
        id = state.getHash()
        if id not in self.estimations:
            self.estimations[id] = 0.5

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
                state.end = 1.0 if self.symbol == 1 else 0.0
                return state.end
            if result == -3:
                state.end = state.end = 1.0 if self.symbol == -1 else 0.0
                return state.end

        sum = np.sum(np.abs(state.data))
        if sum == BOARD_ROWS * BOARD_COLS:
            state.end = 0.5
            return state.end

        state.end = None
        return state.end

    def save(self):
        fw = open('optimal_policy_' + str(self.symbol), 'wb')
        pickle.dump(self.estimations, fw)
        fw.close()

    def load(self):
        fr = open('optimal_policy_' + str(self.symbol),'rb')
        self.estimations = pickle.load(fr)
        fr.close()

# human interface
# input a number to put a chessman
# | 1 | 2 | 3 |
# | 4 | 5 | 6 |
# | 7 | 8 | 9 |
class HummanAgent(Agent):

    def takeAction(self, t_cur = None, t_max = None):
        self.states[-1].show()
        data = int(input("Input your position:")) - 1
        i = data // int(BOARD_COLS)
        j = int(data - i * BOARD_COLS)
        if self.states[-1].data[i,j] != 0:
            return self.takeAction(t_cur, t_max)
        return [i,j]


show = False

class T3Director(Director):
    def __init__(self):
        self.player1 = None
        self.player2 = None
        self.currentPlayer = None
        self.currentState = None

    def reset(self):
        self.currentState = T3State()
        self.player1.reset()
        self.player2.reset()
        self.currentPlayer = None
        self.feedCurrentState()

    def feedCurrentState(self):
        self.player1.feedState(self.currentState)
        self.player2.feedState(self.currentState)

    def play(self, t_cur = None, t_max = None):
        self.reset()
        while True:
            # set current player
            if self.currentPlayer == self.player1:
                self.currentPlayer = self.player2
            else:
                self.currentPlayer = self.player1
            if show:
                self.currentState.show()
            action = self.currentPlayer.takeAction(t_cur, t_max)
            self.currentState, reward = self.currentPlayer.step(action)
            hashValue = self.currentState.getHash()
            self.feedCurrentState()
            isEnd = self.currentPlayer.isEnd()
            if isEnd is None:
                continue
            if isEnd == 0.5:
                self.player1.feedReward(0)
                self.player2.feedReward(0)
                return None
            elif isEnd == 1.0:
                if self.currentPlayer == self.player1:
                    self.player1.feedReward(1)
                    self.player2.feedReward(0)
                else:
                    self.player1.feedReward(0)
                    self.player2.feedReward(1)
                return self.currentPlayer
            else:
                if self.currentPlayer == self.player1:
                    self.player1.feedReward(1)
                    self.player2.feedReward(0)
                    return self.player2
                else:
                    self.player1.feedReward(0)
                    self.player2.feedReward(1)
                    return self.player1

    def train(self, epochs):
        self.player1 = Agent(T3Policy(1))
        self.player2 = Agent(T3Policy(-1))
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
        self.player1.savePolicy()
        self.player2.savePolicy()

    def compete(self, turns):
        self.player1 = Agent(T3Policy(1, 0.1, 0))
        self.player2 = Agent(T3Policy(-1, 0.1, 0))
        p1Win = 0
        p2Win = 0
        self.player1.loadPolicy()
        self.player2.loadPolicy()
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
        self.player1 = HummanAgent(T3Policy(1, 0.1, 0))
        self.player2 = Agent(T3Policy(-1, 0.1, 0))
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

direct = T3Director()
direct.train(2000)
#direct.train(500)
direct.vs_human()