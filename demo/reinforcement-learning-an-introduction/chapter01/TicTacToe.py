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

class T3Environment(Environment):



    #del
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
                #self.winner = 1
                state.end = True
                return state.end
            if result == -3:
                #self.winner = -1
                state.end = True
                return self.end

        # whether it's a tie
        sum = np.sum(np.abs(self.data))
        if sum == BOARD_ROWS * BOARD_COLS:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end