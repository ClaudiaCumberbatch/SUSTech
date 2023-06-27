import numpy as np
import random
import time
import math
import functools
import numba
from numba import jit

infinity = math.inf

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
MAX_DEPTH = 5

# FLIP_WEIGHT = 3
# CERTAIN_WEIGHT = 2
# don't change the class name
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []

        self.FLIP_WEIGHT = 6
        self.CERTAIN_WEIGHT = 3

    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        self.candidate_list, _, terminate_flag = self.get_fair_position(chessboard, self.color)
        if terminate_flag:
            return
        # self.candidate_list.append(self.candidate_list[0])
        # random.shuffle(self.candidate_list)
        global MAX_DEPTH
        if np.sum(chessboard==COLOR_NONE)<8:
            MAX_DEPTH = 8
        elif np.sum(chessboard==COLOR_NONE)<20:
            MAX_DEPTH = 7
        v, move = self.minimax_search(chessboard)
        # print(self.color, "move ", move, " with value = ", v)
        # print(move)
        if move in self.candidate_list:
            self.candidate_list.append(move)

    def get_fair_position(self, chessboard, player):  # return result list, corresponding flip range, whether terminate
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

    def flip(self, chessboard, pos, flips, current_color):  # return翻转之后的棋盘和翻转的棋子个数
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
        return cnt  # 翻转越少越好

    def minimax_search(self, real_chessboard):
        chessboard = real_chessboard.copy()
        global MAX_DEPTH
        """Search game tree to determine best move; return (value, move) pair."""

        def compare_position(result1, result2):
            pos1 = result1[0]
            pos2 = result2[0]
            a = abs(pos1[0] - 3.5) + abs(pos1[1] - 3.5)
            b = abs(pos2[0] - 3.5) + abs(pos2[1] - 3.5)
            return b - a

        def max_value(real_chessboard, current_depth, alpha, beta, last_flip_cnt):
            current_depth += 1
            chessboard = real_chessboard.copy()
            fair_positions, result_and_flip_range, terminate_flag = self.get_fair_position(chessboard, self.color)
            global_terminate_flag = False
            if terminate_flag:
                _, _, global_terminate_flag = self.get_fair_position(chessboard, -self.color)
            if current_depth > MAX_DEPTH or global_terminate_flag:
                return utility(self.color, chessboard) + last_flip_cnt * self.FLIP_WEIGHT, None
            v, move = -infinity, None
            if terminate_flag and (not global_terminate_flag):
                v2, _ = min_value(chessboard, current_depth, alpha, beta, last_flip_cnt)
                if v2 > v: v = v2+300
                if v >= beta: return v, move
                alpha = max(alpha, v)
            result_and_flip_range.sort(key=functools.cmp_to_key(compare_position))
            # random.shuffle(result_and_flip_range) # 尽量选择行列在中间的位置先遍历：下标与3.5的绝对值越接近的越先遍历
            for raf in result_and_flip_range:
                if current_depth == 1 and len(result_and_flip_range)>1:
                    action = raf[0]
                    if ((action[0]==0 and action[1]==0) or (action[0]==0 and action[1]==7) or (action[0]==7 and action[1]==0) or (action[0]==7 and action[1]==7)):
                        continue
                a = raf[0]
                flips = raf[1]
                new_chessboard = chessboard.copy()
                flip_cnt = last_flip_cnt + self.flip(new_chessboard, a, flips, self.color)
                # print(player)
                # print(a)
                # print(chessboard)
                v2, _ = min_value(new_chessboard, current_depth, alpha, beta, flip_cnt)
                if v2 > v:
                    move = a
                    v = v2
                if v >= beta: return v, move
                alpha = max(alpha, v)
            return v, move

        def min_value(real_chessboard, current_depth, alpha, beta, last_flip_cnt):
            current_depth += 1
            chessboard = real_chessboard.copy()
            fair_positions, result_and_flip_range, terminate_flag = self.get_fair_position(chessboard, -self.color)
            global_terminate_flag = False
            if terminate_flag:
                _, _, global_terminate_flag = self.get_fair_position(chessboard, self.color)
            if current_depth > MAX_DEPTH or global_terminate_flag:
                return utility(self.color, chessboard) + last_flip_cnt * self.FLIP_WEIGHT, None
            v, move = infinity, None
            if terminate_flag and (not global_terminate_flag):
                # if self.is_terminal(chessboard, -self.color) and (not self.is_terminal(chessboard, self.color)):
                v2, _ = max_value(chessboard, current_depth, alpha, beta, last_flip_cnt)
                if v2 < v: v = v2-300
                if v <= alpha: return v, move
                beta = min(beta, v)
            result_and_flip_range.sort(key=functools.cmp_to_key(compare_position))
            # random.shuffle(result_and_flip_range)
            for raf in result_and_flip_range:
                a = raf[0]
                flips = raf[1]
                new_chessboard = chessboard.copy()
                flip_cnt = last_flip_cnt - self.flip(new_chessboard, a, flips, -self.color)
                # print(player)
                # print(a)
                # print(chessboard)

                v2, _ = max_value(new_chessboard, current_depth, alpha, beta, flip_cnt)
                if v2 < v:
                    move = a
                    v = v2
                if v <= alpha: return v, move
                beta = min(beta, v)
            return v, move

        return max_value(chessboard, 0, -infinity, infinity, 0)

@jit(nopython=True)
def utility(player, chessboard):
    '''
    weight = [[1,8,3,7,7,3,8,1],
              [8,3,2,5,5,2,3,8],
              [3,2,6,6,6,6,2,3],
              [7,5,6,4,4,6,5,7],
              [7,5,6,4,4,6,5,7],
              [3,2,6,6,6,6,2,3],
              [8,3,2,5,5,2,3,8],
              [1,8,3,7,7,3,8,1]]
    '''
    weight = [[-300, 100, -17, -15, -15, -17, 100, -300],
              [100, 65, -10, -8, -8, -10, 65, 100],
              [-17, -10, -9, -6, -6, -9, -9, -10, -17],
              [-15, -8, -6, -3, -3, -6, -8, -15],
              [-15, -8, -6, -3, -3, -6, -8, -15],
              [-17, -10, -9, -6, -6, -9, -9, -10, -17],
              [100, 65, -10, -8, -8, -10, 65, 100],
              [-300, 100, -17, -15, -15, -17, 100, -300]]
    cnt = 0
    cnt2 = 0
    for r in range(8):
        for c in range(8):
            if chessboard[r][c] == player:
                cnt += weight[r][c]  # smaller
                if certain_decide(chessboard, (r, c)):
                    # cnt += weight[r][c] * 3
                    cnt -= 20
                    # cnt += weight[r][c] * self.CERTAIN_WEIGHT
            elif chessboard[r][c] == -player:
                cnt2 += weight[r][c]  # bigger
                if certain_decide(chessboard, (r, c)):
                    # cnt2 += weight[r][c] * 3
                    cnt2 -= 20
    return cnt-cnt2

@jit(nopython=True)
def certain_decide(chessboard, pos):  # position是不是确定子,player由当前pos上棋子的颜色确定
    dir1 = [(-1, -1), (1, 1)]
    dir2 = [(-1, 0), (1, 0)]
    dir3 = [(-1, 1), (1, -1)]
    dir4 = [(0, 1), (0, -1)]
    flag1 = flag2 = flag3 = flag4 = False
    player = chessboard[pos[0]][pos[1]]
    cur_pos = [pos[0], pos[1]]
    for d in dir1:
        cur_pos[0] = d[0] + pos[0]
        cur_pos[1] = d[1] + pos[1]
        if cur_pos[0] < 0 or cur_pos[0] > 7 or cur_pos[1] < 0 or cur_pos[1] > 7:
            flag1 = True
            break
        if (chessboard[cur_pos[0]][cur_pos[1]] == COLOR_NONE or chessboard[cur_pos[0]][
            cur_pos[1]] == -player):
            continue
        while (cur_pos[0] >= 0 and cur_pos[0] <= 7 and cur_pos[1] >= 0 and cur_pos[1] <= 7 and
               chessboard[cur_pos[0]][cur_pos[1]] == player):
            cur_pos[0] += d[0]
            cur_pos[1] += d[1]
        if cur_pos[0] < 0 or cur_pos[0] > 7 or cur_pos[1] < 0 or cur_pos[1] > 7:
            flag1 = True
            break
    for d in dir2:
        cur_pos[0] = d[0] + pos[0]
        cur_pos[1] = d[1] + pos[1]
        if cur_pos[0] < 0 or cur_pos[0] > 7 or cur_pos[1] < 0 or cur_pos[1] > 7:
            flag2 = True
            break
        if (chessboard[cur_pos[0]][cur_pos[1]] == COLOR_NONE or chessboard[cur_pos[0]][
            cur_pos[1]] == -player):
            continue
        while (cur_pos[0] >= 0 and cur_pos[0] <= 7 and cur_pos[1] >= 0 and cur_pos[1] <= 7 and
               chessboard[cur_pos[0]][cur_pos[1]] == player):
            cur_pos[0] += d[0]
            cur_pos[1] += d[1]
        if cur_pos[0] < 0 or cur_pos[0] > 7 or cur_pos[1] < 0 or cur_pos[1] > 7:
            flag2 = True
            break
    for d in dir3:
        cur_pos[0] = d[0] + pos[0]
        cur_pos[1] = d[1] + pos[1]
        if cur_pos[0] < 0 or cur_pos[0] > 7 or cur_pos[1] < 0 or cur_pos[1] > 7:
            flag3 = True
            break
        if (chessboard[cur_pos[0]][cur_pos[1]] == COLOR_NONE or chessboard[cur_pos[0]][
            cur_pos[1]] == -player):
            continue
        while (cur_pos[0] >= 0 and cur_pos[0] <= 7 and cur_pos[1] >= 0 and cur_pos[1] <= 7 and
               chessboard[cur_pos[0]][cur_pos[1]] == player):
            cur_pos[0] += d[0]
            cur_pos[1] += d[1]
        if cur_pos[0] < 0 or cur_pos[0] > 7 or cur_pos[1] < 0 or cur_pos[1] > 7:
            flag3 = True
            break
    for d in dir4:
        cur_pos[0] = d[0] + pos[0]
        cur_pos[1] = d[1] + pos[1]
        if cur_pos[0] < 0 or cur_pos[0] > 7 or cur_pos[1] < 0 or cur_pos[1] > 7:
            flag4 = True
            break
        if (chessboard[cur_pos[0]][cur_pos[1]] == COLOR_NONE or chessboard[cur_pos[0]][
            cur_pos[1]] == -player):
            continue
        while (cur_pos[0] >= 0 and cur_pos[0] <= 7 and cur_pos[1] >= 0 and cur_pos[1] <= 7 and
               chessboard[cur_pos[0]][cur_pos[1]] == player):
            cur_pos[0] += d[0]
            cur_pos[1] += d[1]
        if cur_pos[0] < 0 or cur_pos[0] > 7 or cur_pos[1] < 0 or cur_pos[1] > 7:
            flag4 = True
            break
    # if flag1 and flag2 and flag3 and flag4:
    #     print(chessboard)
    #     print(pos, "is a certain point")
    return flag1 and flag2 and flag3 and flag4
