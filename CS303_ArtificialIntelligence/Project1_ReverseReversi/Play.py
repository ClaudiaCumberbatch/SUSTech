import numpy as np
import qyz as ai
import project1_templet as zsc
import greedy

# AIb = greedy.AI(8, ai.COLOR_BLACK, 1)
# AIw = zsc.AI(8, ai.COLOR_WHITE, 1)
AIb = zsc.AI(8, ai.COLOR_BLACK, 1)
AIw = zsc.AI(8, ai.COLOR_WHITE, 1)
currentColor = ai.COLOR_BLACK

# board = np.zeros((8, 8))
# board[3][3] = board[4][4] = ai.COLOR_WHITE
# board[3][4] = board[4][3] = ai.COLOR_BLACK

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

def play(black_flip_w, black_certain_w, white_flip_w, white_certain_w):
    print("black flip =", black_flip_w, "black certain =", black_certain_w)
    print("white flip =", white_flip_w, "white certain =", white_certain_w)
    print("game running")
    board = np.zeros((8, 8))
    board[3][3] = board[4][4] = ai.COLOR_WHITE
    board[3][4] = board[4][3] = ai.COLOR_BLACK
    AIb.FLIP_WEIGHT = black_flip_w
    AIb.CERTAIN_WEIGHT = black_certain_w
    AIw.FLIP_WEIGHT = white_flip_w
    AIw.CERTAIN_WEIGHT = white_certain_w
    currentColor = ai.COLOR_BLACK
    while ai.COLOR_NONE in board:
        # print(board)
        if currentColor == ai.COLOR_BLACK:
            AIb.go(board)
            # print(AIb.candidate_list)
            if len(AIb.candidate_list) > 0:
                work(AIb.candidate_list[len(AIb.candidate_list) - 1], currentColor, board)
            currentColor = ai.COLOR_WHITE
        else:
            AIw.go(board)
            # print(AIw.candidate_list)
            if len(AIw.candidate_list) > 0:
                work(AIw.candidate_list[len(AIw.candidate_list) - 1], currentColor, board)
            currentColor = ai.COLOR_BLACK

    # print(board)

    cnt_W=np.sum(board==ai.COLOR_WHITE)
    cnt_B=np.sum(board==ai.COLOR_BLACK)

    print("white score = ", cnt_W, " and black score = ", cnt_B)
    if cnt_W<cnt_B:
        print("White wins!")
    if cnt_B<cnt_W:
        print("Black wins!")
    if cnt_B==cnt_W:
        print("DRAW!")

# play(1,1,1,1)
# for i in range(2,5):
#     for j in range(i, 5):
#         if i==2 and (j == 2 or j == 3 or j == 4): continue
#         play(i,j,i+1,j+1)
#         play(i+1,j+1,i,j)
play(5,5,4,4)