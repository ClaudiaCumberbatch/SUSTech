import numpy as np
import qyz as ai
import project1_templet as zsc
import greedy
import monte_carlo as mc

# AIb = mc.AI(8, ai.COLOR_BLACK, 1)
AIw = zsc.AI(8, ai.COLOR_WHITE, 1)
AIb = greedy.AI(8, ai.COLOR_BLACK, 1)
# AIw = zsc.AI(8, ai.COLOR_WHITE, 1)
currentColor = ai.COLOR_BLACK

board = np.zeros((8, 8))
board[3][3] = board[4][4] = ai.COLOR_WHITE
board[3][4] = board[4][3] = ai.COLOR_BLACK


def change(pos, dir, color, board):
    pos = (pos[0] + dir[0], pos[1] + dir[1])
    while ai.in_board(pos):
        if board[pos] == color:
            return
        board[pos] = color
        pos = (pos[0] + dir[0], pos[1] + dir[1])
    return


def work(pos, color, board):
    board[pos] = color
    for i in range(8):
        if ai.fd(board, pos, i, color):
            change(pos, ai.NT[i], color, board)
    return


while ai.COLOR_NONE in board:
    print(board)
    if currentColor == ai.COLOR_BLACK:
        AIb.go(board)
        print(AIb.candidate_list)
        if len(AIb.candidate_list) > 0:
            work(AIb.candidate_list[len(AIb.candidate_list) - 1], currentColor, board)
        currentColor = ai.COLOR_WHITE
    else:
        AIw.go(board)
        # print(AIw.candidate_list)
        if len(AIw.candidate_list) > 0:
            work(AIw.candidate_list[len(AIw.candidate_list) - 1], currentColor, board)
        currentColor = ai.COLOR_BLACK

    print(board)

cnt_W = np.sum(board == ai.COLOR_WHITE)
cnt_B = np.sum(board == ai.COLOR_BLACK)

print("white score = ", cnt_W, " and black score = ", cnt_B)
if cnt_W < cnt_B:
    print("White wins!")
if cnt_B < cnt_W:
    print("Black wins!")
if cnt_B == cnt_W:
    print("DRAW!")
