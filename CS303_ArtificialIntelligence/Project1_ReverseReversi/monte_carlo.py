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
t1=0
exploration = True

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

    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        global t1
        t1 = time.time()
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        self.candidate_list, raf, terminate_flag = get_fair_position(chessboard, self.color)
        if terminate_flag:
            return
        v, move = monte_carlo_tree_search(chessboard, self.color)
        # print("move ", move, " with value = ", v)
        # print(move)
        if move in self.candidate_list:
            self.candidate_list.append(move)

# @jit(nopython=True)
def get_fair_position(chessboard, player):  # return result list, corresponding flip range, whether terminate
    # self.candidate_list.clear()
    # ==================================================================
    # Write your algorithm here
    # Here is the simplest sample:Random decision
    idx = np.where(chessboard == COLOR_NONE)
    idx = list(zip(idx[0], idx[1]))  # empty position
    if len(idx) == 0: return [], [], True
    global exploration
    if len(idx) < 30: exploration = False
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

# @jit(nopython=True)
def flip(chessboard, pos, flips, current_color):  # return翻转之后的棋盘和翻转的棋子个数
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
    # print(utility(chessboard))
    return cnt  # 翻转越少越好

class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states."""

    def __init__(self, player, parent=None, state=None, U=0, N=0):
        self.__dict__.update(player = player, parent=parent, state=state, U=U, N=N)
        self.children = {}  # dict of {node: action} or {actione: node}
        self.actions = None

# @jit(nopython=True)
def ucb(n, C=1.4):
    # return np.inf if n.N == 0 else (1 - n.U / n.N) + C * np.sqrt(np.log(n.parent.N) / n.N)
    return np.inf if n.N == 0 else ucb_normal(n.U, n.N, n.parent.N)

@jit(nopython=True)
def ucb_normal(U, N, pN):
    global exploration
    if exploration: C = 1.4
    else: C = 0.8
    return  (1 - U / N) + C * np.sqrt(np.log(pN) / N)

@jit(nopython=True)
def utility(chessboard, player):
    cnt = 0
    cnt2 = 0
    '''
    weight = [[1, 8, 3, 7, 7, 3, 8, 1],
              [8, 3, 2, 5, 5, 2, 3, 8],
              [3, 2, 6, 6, 6, 6, 2, 3],
              [7, 5, 6, 4, 4, 6, 5, 7],
              [7, 5, 6, 4, 4, 6, 5, 7],
              [3, 2, 6, 6, 6, 6, 2, 3],
              [8, 3, 2, 5, 5, 2, 3, 8],
              [1, 8, 3, 7, 7, 3, 8, 1]]
    '''
    weight = [[-1000, 65, -17, -15, -15, -17, 65, -1000],
              [65, 35, -10, -8, -8, -10, 35, 65],
              [-17, -10, -9, -6, -6, -9, -9, -10, -17],
              [-15, -8, -6, -3, -3, -6, -8, -15],
              [-15, -8, -6, -3, -3, -6, -8, -15],
              [-17, -10, -9, -6, -6, -9, -9, -10, -17],
              [65, 35, -10, -8, -8, -10, 35, 65],
              [-1000, 65, -17, -15, -15, -17, 65, -1000]]
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == player: cnt += weight[i][j]
            elif chessboard[i][j] == -player: cnt2 += weight[i][j]
    '''
    for row in chessboard:
        for e in row:
            if e == player: cnt += 1
            elif e == -player: cnt2 += 1
    '''
    if cnt > cnt2: return 1
    else: return -1

def monte_carlo_tree_search(state, player, N=10000): # state即chessboard
    def select(n):
        """
        select a leaf node in the tree according to their UCB values
        """
        while (n.children):
            n = max(n.children.keys(), key=ucb)
        return n
        # if n.children:
        #     return select(max(n.children.keys(), key=ucb))
        # else:
        #     return n

    def expand(n):
        """ 
        expand the leaf node by adding all its children states (FOLLOW-LINK)
        """
        result, raf, terminate_flag = get_fair_position(n.state, player)
        if not n.children and not terminate_flag:
            for e in raf:
                pos = e[0]
                flips = e[1]
                new_state = state.copy()
                flip(new_state, pos, flips, player)
                nd = MCT_Node(player = -n.player, parent=n, state=new_state)
                # nd = MCT_Node(state=game.result(n.state, action), parent=n)
                n.children[nd] = pos

        return select(n)

    def simulate(state, local_player):
        def compare_position(result1, result2):
            pos1 = result1[0]
            pos2 = result2[0]
            a = abs(pos1[0] - 3.5) + abs(pos1[1] - 3.5)
            b = abs(pos2[0] - 3.5) + abs(pos2[1] - 3.5)
            return b - a
        """ 
        simulate the utility of current state using a random strategy
        """
        # player = state.to_move
        result, raf, terminate_flag = get_fair_position(state, local_player)
        search_depth = 0
        new_state = state.copy()
        # while not game.is_terminal(state):
        while (not terminate_flag) and search_depth < 20:
        # while (not terminate_flag):
            search_depth += 1
            raf.sort(key=functools.cmp_to_key(compare_position))
            # random.shuffle(raf)
            e = raf[0]
            action = e[0]
            flip(new_state, action, e[1], local_player)
            local_player = -local_player
            _, raf, terminate_flag = get_fair_position(new_state, local_player)

        v = utility(state, player)
        return v

    def backprop(n, utility):
        """
        passing the utility back to all parent nodes
        """
        while n:
            if utility > 0: n.U += utility
            n.N += 1
            n = n.parent
            utility = -utility

        # if n.parent:
        #     backprop(n.parent, -utility)

    root = MCT_Node(player=player, state=state)

    for _ in range(N):
        # PLAYOUT
        t2 = time.time()
        global t1
        if t2 - t1 > 4: break
        leaf = select(root)
        child = expand(leaf)
        result = simulate(child.state, -player)
        backprop(child, result)

    # TODO: select the action
    v, move = 0.000, (0, 0)
    for node, action in root.children.items():
        # print(action, node.U, node.N)
        if len(root.children.items())>1 and ((action[0]==0 and action[1]==0) or (action[0]==0 and action[1]==7) or (action[0]==7 and action[1]==0) or (action[0]==7 and action[1]==7)):
            continue
        if node.N == 0: continue
        if node.U / node.N > v:
            v = node.U / node.N
            move = action
    #     print(v)
    return v, move
