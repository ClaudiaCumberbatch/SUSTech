import numpy as np
import random
import math
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
NT = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def in_board(pos):
    return (0 <= pos[0] and pos[0] <= 7) and (0 <= pos[1] and pos[1] <= 7)


def fd(chessboard, pos, id, color):
    pos = (pos[0] + NT[id][0], pos[1] + NT[id][1])
    while in_board(pos):
        if chessboard[pos] == color:
            return True
        if chessboard[pos] == COLOR_NONE:
            return False
        pos = (pos[0] + NT[id][0], pos[1] + NT[id][1])
    return False


def change(board, pos, dir, color):
    pos = (pos[0] + dir[0], pos[1] + dir[1])
    while in_board(pos):
        if board[pos] == color:
            return
        board[pos] = color
        pos = (pos[0] + dir[0], pos[1] + dir[1])
    return


def work(chessboard, pos, color):
    board = np.copy(chessboard)
    board[pos] = color
    for i in range(8):
        if fd(board, pos, i, color):
            change(board, pos, NT[i], color)
    return board


#don't change the class name
class AI(object):
    #chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        #You are white or black
        self.color = color
        #the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []

    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        #==================================================================
        #Write your algorithm here
        #Here is the simplest sample:Random decision
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        for pos in idx:
            for id in range(8):
                if in_board(
                    (pos[0] + NT[id][0], pos[1] +
                     NT[id][1])) and chessboard[(pos[0] + NT[id][0], pos[1] +
                                                 NT[id][1])] == (-self.color):
                    result = fd(chessboard, pos, id, self.color)
                    if result:
                        self.candidate_list.append(pos)
                        break
        if len(self.candidate_list) == 0:
            return
        self.candidate_list.append(self.candidate_list[0])
        #==============Find new pos========================================
        # Make sure that the position of your decision on the chess board is empty.
        # If not, the system will return error.ƒ
        # Add your decision into candidate_list, Records the chessboard
        # You need to add all the positions which are valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        #candidate_list example: [(3,3),(4,4),(4,4)]
        # we will pick the last element of the candidate_list as the position you choose.
        #In above example, we will pick (4,4) as your decision.
        # If there is no valid position, you must return an empty


infinity = math.inf

