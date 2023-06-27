import numpy as np
import random
import time
import math
import functools
infinity = math.inf

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
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
        self.candidate_list, raf, terminate_flag = self.get_fair_position(chessboard, self.color)
        if terminate_flag:
            return
        min_flip = infinity
        move = (0, 0)
        for a in raf:
            m = a[0]
            f = a[1]
            new_chessboard = chessboard.copy()
            flip = self.flip(new_chessboard, m, f, self.color)
            if min_flip > flip:
                min_flip = flip
                move = m
        # print("move ", move, " with value = ", v)
        # print(move)
        if move in self.candidate_list:
            self.candidate_list.append(move)


    def get_fair_position(self, chessboard, player): #return result list, corresponding flip range, whether terminate
        # self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))  # empty position
        if len(idx) == 0: return [], [], False
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        result = []
        result_and_flip_range = []
        for pos in idx:
            flips = []
            fair_flag = False
            for direction in directions:
                cur_pos = [0, 0]
                cur_pos[0] = pos[0] + direction[0]
                cur_pos[1] = pos[1] + direction[1]
                if cur_pos[0] < 0 or cur_pos[0] > 7 or cur_pos[1] < 0 or cur_pos[1] > 7:
                    continue
                if (chessboard[cur_pos[0]][cur_pos[1]] == COLOR_NONE or chessboard[cur_pos[0]][
                    cur_pos[1]] == player):
                    continue
                while (cur_pos[0] >= 0 and cur_pos[0] <= 7 and cur_pos[1] >= 0 and cur_pos[1] <= 7 and
                       chessboard[cur_pos[0]][cur_pos[1]] == -player):
                    cur_pos[0] += direction[0]
                    cur_pos[1] += direction[1]
                if cur_pos[0] < 0 or cur_pos[0] > 7 or cur_pos[1] < 0 or cur_pos[1] > 7:
                    continue
                if chessboard[cur_pos[0]][cur_pos[1]] == player:
                    fair_flag = True
                    flips.append(cur_pos)
            if fair_flag:
                result.append(pos)
                raf = [pos, flips]
                result_and_flip_range.append(raf)
                # flip_range.append(flips)
                # print(pos, flips)
        terminate_flag = False
        if len(result) == 0: terminate_flag = True
        # print(result,"\n", flip_range)
        # print(len(result), len(flip_range))
        return result, result_and_flip_range, terminate_flag

    def flip(self, chessboard, pos, flips, current_color): # return翻转之后的棋盘和翻转的棋子个数
        # print(chessboard, pos, flips, current_color)
        cnt = 0
        for cur_pos in flips:
            if pos[0] == cur_pos[0]:
                chessboard[pos[0], min(pos[1], cur_pos[1]): max(pos[1], cur_pos[1]) + 1] = current_color
                cnt += abs(pos[1] - cur_pos[1])
            elif pos[1] == cur_pos[1]:
                chessboard[min(pos[0], cur_pos[0]): max(pos[0], cur_pos[0]) + 1, pos[1]] = current_color
                cnt += abs(pos[0] - cur_pos[0])
            else:
                cnt += abs(pos[0] - cur_pos[0])
                row_flag = 1 if cur_pos[0] > pos[0] else -1
                col_flag = 1 if cur_pos[1] > pos[1] else -1
                lo = min(cur_pos[0], pos[0])
                hi = max(cur_pos[0], pos[0])
                for i in range(hi - lo + 1):
                    chessboard[pos[0] + i * row_flag][pos[1] + i * col_flag] = current_color
        return cnt # 翻转越少越好

