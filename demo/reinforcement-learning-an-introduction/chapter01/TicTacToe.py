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

class T3Policy(Policy):

    def __init__(self, symbol, stepSize = 0.1, exploreRate = 0.1):
        self.symbol = symbol
        self.stepSize = stepSize
        self.exploreRate = exploreRate
        self.estimations = dict()

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
            return self.end

        state.end = None
        return state.end

class T3Agent(Agent):




