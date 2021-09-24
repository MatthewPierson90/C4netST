# -*- coding: utf-8 -*-
"""
@author: Matthew

Everything needed for minimax and more.
"""


import time
tt = time.time
import numpy as np
from numba import njit,prange




@njit(nogil=True)
def new_board():
    """
    creates a new game board. Empty spaces are represented with 0's,
    player 1's pieces will be represented with 1s, 
    player 2's pieces will be represented with 2s.
    """
    board = np.zeros((7,6))
    return(board)



@njit(nogil=True) 
def available_moves(board):
    """
    Determines all available moves given a game board.
    Input: a game board
    Output: an np.array with a 1 in the ith position if i is an available move.
    i = 0 corresponds to column 1. 
    """
    moves = np.zeros(7)
    for n in range(7):
        if board[n][0]==0:
            moves[n]=1
    return(moves.astype(np.int32))


@njit(nogil=True) 
def make_move(board, col, player):
    """
    makes a move on a game board.  
    Input: the game board to make a move on, 
           a chosen move/ column  (1 through 7), 
           the player making the move
    output: the new game board
    """
    # print('in make move')
    for n in range(6):
        if n == 5:
            board[col-1][n] = player
            break
        elif board[col-1][n+1]!= 0:
            board[col-1][n] = player
            break
    return(board)



@njit(nogil=True) 
def lst_to_str(lst):
    """
    Turns a list (which is a representation of a specific row/col/diagonal) into a string. 
    Will be used to calculate the current board score.
    Input: a list or np.array with 0,1, or 2 as entries
    output: a string with blanks (for 0s), Xs (for 1s), and Os (for 2s)
    """
    lst_str = ''
    for item in lst:
        if item == 0:
            lst_str+= ' '
        elif item == 1:
            lst_str+='X'
        else:
            lst_str+='O'
    return lst_str



@njit(nogil=True) 
def row_n_as_str(board,n):
    """
    Turns a specified row for a game board into a string. 
    Will be used to determine if the game is over, and which player won.
    Input: a gameboard and row
    output: a string with blanks (for 0s), Xs (for 1s), and Os (for 2s)
    """
    row_str = ''
    for m in range(7):
        if board[m][n]==0:
            row_str+=' '
        elif board[m][n]==1:
            row_str+='X'
        else:
            row_str+='O'
    return(row_str)




@njit(nogil=True) 
def col_n_as_str(board,n):
    """
    Turns a specified column for a game board into a string. 
    Will be used to determine if the game is over, and which player won.
    Input: a gameboard and column
    output: a string with blanks (for 0s), Xs (for 1s), and Os (for 2s)
    """
    col_str = ''
    for s in board[n]:
        if s == 0:
            col_str+=' '
        elif s==1:
            col_str+='X'
        else:
            col_str+='O'
    return(col_str)



@njit(nogil=True) 
def minor_diag_n_as_str(board,n):
    """
    Turns a specified diagonal for a game board into a string. 
    Will be used to determine if the game is over, and which player won.
    Input: a gameboard and diagonal
    output: a string with blanks (for 0s), Xs (for 1s), and Os (for 2s)
    """
    diag_str=''
    for p in range(7):
      for q in range(6):
        if p+q==n:
            if board[p][q] == 0:
                diag_str+=' '
            elif board[p][q]==1:
                diag_str+='X'
            else:
                diag_str+='O'
    return(diag_str)




@njit(nogil=True) 
def major_diag_n_as_str(board,n):
    """
    Turns a specified diagonal for a game board into a string. 
    Will be used to determine if the game is over, and which player won.
    Input: a gameboard and diagonal
    output: a string with blanks (for 0s), Xs (for 1s), and Os (for 2s)
    """
    diag_str=''
    for p in range(7):
      for q in range(6):
        if p-q==n:
            if board[p][q] == 0:
                diag_str+=' '
            elif board[p][q]==1:
                diag_str+='X'
            else:
                diag_str+='O'
    return(diag_str)




@njit(nogil=True, cache=True) 
def has_won(board,player):
    """
    Determines if a specified player has won
    inputs: game board, player
    outputs: True/ False
    """
    if player ==1:
        four = 'XXXX'
    else:
        four ='OOOO'
    for n in range(6):
        row_str = row_n_as_str(board,n)
        if four in row_str:
            return(True)
    for n in range(3,9):
        if four in minor_diag_n_as_str(board,n):
            return(True)
    for n in range(-4,3):
        if four in major_diag_n_as_str(board,n):
            return(True)
    for n in range(7):
        if four in col_n_as_str(board,n):
            return(True)
    return(False)



@njit(nogil=True, cache=True) 
def has_won_z(board):
    """
    Determines if either player has won
    inputs: game board
    outputs: 0 (if neither won), 1 if X won, 2 if O won
    """
    fourx = 'XXXX'
    fouro ='OOOO'
    for n in range(6):
        row_str = row_n_as_str(board,n)
        if fourx in row_str:
            return(1)
        elif fouro in row_str:
            return(2)
    for n in range(3,9):
        mrd_str = minor_diag_n_as_str(board,n)
        if fourx in mrd_str:
            return(1)
        elif fouro in mrd_str:
            return(2)
    for n in range(-4,3):
        mjd_str=major_diag_n_as_str(board,n)
        if fourx in mjd_str:
            return(1)
        elif fouro in mjd_str:
            return 2
    for n in range(7):
        col_str= col_n_as_str(board,n)
        if fourx in col_str:
            return(1)
        elif fouro in col_str:
            return 2
    return(0)




@njit(nogil=True, cache=True) 
def is_game_over(board):
    """
    Determines if the game is over.
    inputs: board
    outputs: True/ false
    """
    moves = max(available_moves(board))
    if  moves == 0:
        return True
    p1w = has_won(board,1)
    p2w = has_won(board,2)
    if p1w == True:
        return True
    elif  p2w == True:
        return True
    else:
        return False



@njit(nogil=True) 
def lst_to_num(lst):
    """
    Uniquely identifies a list of 0's 1's and 2's with a number. 
    The list must have a length less than or equal to 7.
    Will be used in the board score calculation.
    input: list
    output: identifying number
    """
    x_plst = np.array([2,3,5,7,11,13,17])
    o_plst = np.array([19,23,29,31,37,41,43])
    #lst is a row, column, or diagonal
    num = 1
    for n in range(len(lst)):
        if lst[n]==1:
            num*=x_plst[n]
        elif lst[n] == 2:
            num*=o_plst[n]
    return(num)


@njit(nogil=True) 
def lst_score(row, three_mult=1.5):
    
    """
    Turns a list of numbers into a score.  The list of numbers can represent 
    a row, column, or diagonal (not just a row).  
    The function does this by determining the
    number of consecutive pieces for both players, 
    and scaling 3 consecutive pieces by a specified amount.
    inputs: a list representing a row, column, or diagonal, and a 3 consecutive piece multiplier.
    outputs: a score for the list.
    """
    two=['ZZ  ','Z Z ','Z  Z',' ZZ ',' Z Z','  ZZ']
    three=['ZZZ ','ZZ Z','Z ZZ',' ZZZ']
    row_str = lst_to_str(row)
    x_score = 0
    player ='X'
    two_var = [s.replace('Z',player) for s in two]
    three_var = [s.replace('Z',player) for s in three]
    three_var_win = ' '+player*3+' '
    for s in two_var:
        if s in row_str:
            x_score+=1
    for s in three_var:
        if s in row_str:
            x_score+=three_mult
    if three_var_win in row_str:
        x_score+=10
    if 'XXXX' in row_str:
        x_score = 100
    o_score = 0
    player ='O'
    two_var = [s.replace('Z',player) for s in two]
    three_var = [s.replace('Z',player) for s in three]
    three_var_win = ' '+player*3+' '
    for s in two_var:
        if s in row_str:
            o_score+=1
    for s in three_var:
        if s in row_str:
            o_score+=three_mult
    if three_var_win in row_str:
        o_score+=10
    if 'OOOO' in row_str:
        o_score = 100
    return(x_score-o_score)


@njit(nogil=True) 
def make_product_zeros():
    """
    Creates an np.array which will be used in the board score calculation.
    """
    return(np.zeros((2187,7)))


@njit(nogil=True) 
def my_product(n):
    """
    Creates all possible lists of length n with values 0,1,2.
    Used in the board score calculation.
    """
    three = range(3)
    values = np.zeros((2187,7))
    count = 0
    for s7 in three:
        for s6 in three:
            for s5 in three:
                for s4 in three:
                    for s3 in three:
                        for s2 in three:
                            for s1 in three:
                                values[count] = np.array([s1,s2,s3,s4,s5,s6,s7])
                                count+=1
    last = 3**n
    return(values[:last])



@njit(nogil=True) 
def make_n_array(n,three_mult=1.5):
    """
    Creates an array whose first row uniquely identifies 
    a list of length n with values 0,1,2 with a number using lst_to_num,
    and whose second row provides a score for that list.
    This will be used to pre compute scores for rows, columns, and diagonals
    to reduce computation time in the total board score calculation.
    inputs: the length of the list, the 3 consecutive piece multiplier.
    outputs: an np.array with 2 rows
    """
    to = 3**n
    ar = np.zeros((2,to))
    count = 0
    for n1 in range(to):
        lst1 = []
        for m in range(n):
            s1=0
            if my_product(7)[n1][m]==1:
                s1=1
            elif my_product(7)[n1][m] ==2:
                s1=2
            lst1.append(s1)
        lst1 = np.array(lst1)
        ar[0][count]=lst_to_num(lst1)
        ar[1][count]=lst_score(lst1,three_mult)
        count+=1
    return ar

# if __name__=='__main__':
len_4_vals = make_n_array(4)
len_4_vals.flags.writeable = False
len_5_vals = make_n_array(5)
len_5_vals.flags.writeable = False
len_6_vals = make_n_array(6)
len_6_vals.flags.writeable = False
len_7_vals = make_n_array(7)
len_7_vals.flags.writeable = False




@njit(nogil=True, cache=True)
def num_board_to_score(lst,
                       row_mult,
                       col_mult,
                       diag_mult,
                       len_4_vals,
                       len_5_vals,
                       len_6_vals ,
                       len_7_vals ):
    """
    Calculates a score for a board.  
    The board is represented by an np array,
    with each entry corresponding to a specific row, column and  diagonal.
    The score is calculated by multiplying the list score with a multiplier.
    """
    #lst: row, col, maj, min
    nbts_score = 0.0
    # print('in num_board_to_score')
    for n in range(6):
        in_steps1 = np.where(len_7_vals[0]==lst[0][n])
        in_steps2 = in_steps1[0][0]
        to_mult = len_7_vals[1][in_steps2]
        nbts_score+=row_mult*to_mult
        # if abs(score) is above 90 then a player has won and the game is over.
        if nbts_score >= 90 or nbts_score <= -90:
            return(nbts_score)
    for n in range(7):
        to_mult =len_6_vals[1][np.where(len_6_vals[0]==lst[1][n])[0][0]]
        nbts_score+=col_mult*to_mult 
        if nbts_score >= 90 or nbts_score <= -90:
            return(nbts_score)
    for n in range(6):
        if n==0 or n==5:
            to_mult =len_4_vals[1][np.where(len_4_vals[0]==lst[2][n])[0][0]]
            nbts_score+=diag_mult*to_mult 
        elif n==1 or n==4:
            to_mult =len_5_vals[1][np.where(len_5_vals[0]==lst[2][n])[0][0]]
            nbts_score+=diag_mult*to_mult 
        elif n==2 or n == 3:
            to_mult = len_6_vals[1][np.where(len_6_vals[0]==lst[2][n])[0][0]]
            nbts_score+=diag_mult*to_mult 
        if nbts_score >= 90 or nbts_score <= -90:
            return(nbts_score)
    for n in range(6):
        if n==0 or n==5:
            to_mult =len_4_vals[1][np.where(len_4_vals[0]==lst[3][n])[0][0]]
            nbts_score+=diag_mult*to_mult
        elif n==1 or n==4:
            to_mult =len_5_vals[1][np.where(len_5_vals[0]==lst[3][n])[0][0]]
            nbts_score+=diag_mult*to_mult
        elif n==2 or n == 3:
            to_mult =len_6_vals[1][np.where(len_6_vals[0]==lst[3][n])[0][0]]
            nbts_score+=diag_mult*to_mult
        if nbts_score >= 90 or nbts_score <= -90:
            return(nbts_score)
    return(nbts_score)


@njit(nogil=True, cache=True)
def board_to_num(board):
    """
    Turns the board into a 4 by 7 array with each entry
     uniquely identifying a row, column, or diagonal.
    """
    r7 = range(7)
    r6 = range(6)
    col_nums = [lst_to_num(board[m]) for m in r7]
    row_nums = [lst_to_num([board[m][n] for m in r7]) for n in r6]
    minor_diag_nums = []
    major_diag_nums = []
    for n in range(3,9):
        diag = []
        for p in range(7):
            for q in range(6):
              if p+q==n:
                  diag.append(board[p][q])
        minor_diag_nums.append(lst_to_num(diag))
    for n in range(-2,4):
        diag=[]
        for p in range(7):
            for q in range(6):
                if p-q==n:
                    diag.append(board[p][q])
        major_diag_nums.append(lst_to_num(diag))
    the_lst=[row_nums,col_nums,major_diag_nums,minor_diag_nums]
    to_return = np.ones((4,7))
    for n in range(4):
        for m in range(len(the_lst[n])):
            to_return[n][m]=the_lst[n][m]
    return(to_return)




@njit(nogil=True, cache=True)
def board_eval(board,
               row_mult,
               col_mult,
               diag_mult,
               len_4_vals,
               len_5_vals,
               len_6_vals,
               len_7_vals):
    """
    Evaluates a given board to determine which player is ahead.
    """
    board_nums = board_to_num(board)
    score = num_board_to_score(board_nums,
                              row_mult,
                              col_mult,
                              diag_mult,
                              len_4_vals,
                              len_5_vals,
                              len_6_vals,
                              len_7_vals)
    return score



@njit(nogil=True, cache=True) 
def copy_board(board):
    """
    makes a copy of a game board.
    """
    copy = new_board()
    for n in range(7):
        for m in range(6):
            copy[n][m]=board[n][m]
    return(copy)



@njit(nogil=True, cache=True) 
def minimax(board,  #game board
            is_max, # True for player 1/X, False for player 2/O
            depth, #how many moves the algorithm will look ahead, changes as algorithm looks ahead
            start_depth, #the original number of moves the algorithm will look ahead, stays constant
            row_mult, #how much the algorithm prioritizes rows
            col_mult, #how much the algorithm prioritizes columns
            diag_mult, #how much the algorithm prioritizes diagonals
            prune, #set to True for alpha beta pruning
            alpha, #alpha for alpha beta pruning
            beta, #beta for alpha beta pruning
            len_4_vals, #precomputed diagonal scores for length 4 diagonals
            len_5_vals, #precomputed diagonal scores for length 5 diagonals
            len_6_vals, #precomputed diagonal+row scores for length 6 diagonals and rows
            len_7_vals): #precomputed diagonal+column scores for length 7 diagonals and columns
    """
    Does the minimax calculation to determine what move a player should make.  
    """
    if is_game_over(board) or depth == 0:
        value = board_eval(board, 
                         row_mult,
                         col_mult,
                         diag_mult,
                         len_4_vals,
                         len_5_vals,
                         len_6_vals,
                         len_7_vals)
        return [value, 8,alpha,beta,0,0,0,0,0,0,0]
    best_move = 8
    if is_max:
        best_value = -1000
        symbol = 1
    else:
        best_value = 1000
        symbol = 2
    moves = available_moves(board)
    if depth == start_depth:
        if is_max:
            value_lst = -6666*np.ones(7)
        else:
            value_lst = 6666*np.ones(7)
    for k in range(len(moves)):
        if moves[k]==0:
            continue
        else:
            board_copy = copy_board(board)
            make_move(board_copy,k+1, symbol)
            if depth == start_depth:
                is_won = has_won_z(board_copy)
                if is_won == 1:
                    best_value = 1000
                    best_move = k+1
                    nvalue_lst = np.ones(7)*-6666
                    nvalue_lst[k] = 1000
                    return [best_value, best_move,alpha,beta,nvalue_lst[0],nvalue_lst[1],nvalue_lst[2],nvalue_lst[3],nvalue_lst[4],nvalue_lst[5],nvalue_lst[6]]
                    break
                elif is_won==2:
                    best_value = -1000
                    best_move = k+1
                    nvalue_lst = np.ones(7)*6666
                    nvalue_lst[k] = -1000
                    return [best_value, best_move,alpha,beta,nvalue_lst[0],nvalue_lst[1],nvalue_lst[2],nvalue_lst[3],nvalue_lst[4],nvalue_lst[5],nvalue_lst[6]]
                    break
            hypothetical_value = minimax(board_copy,  
                                         not is_max, 
                                         depth - 1,
                                         start_depth,
                                         row_mult,
                                         col_mult,
                                         diag_mult, 
                                         True,
                                         alpha, 
                                         beta,
                                         len_4_vals,
                                         len_5_vals,
                                         len_6_vals,
                                         len_7_vals)[0]
            if depth == start_depth:
                value_lst[k]=hypothetical_value
            if is_max == True and hypothetical_value > best_value:
                best_value = hypothetical_value
                alpha = max(alpha, best_value)
                best_move = k+1
            if is_max == False and hypothetical_value < best_value:
                best_value = hypothetical_value
                beta = min(beta, best_value)
                best_move = k+1
            if prune:
                if alpha>=beta and depth!=start_depth:
                    break
        if best_move == 8:
            best_move = np.where(available_moves(board)==1)[0][0]
    if depth!= start_depth:
        return [best_value, best_move,alpha,beta,0,0,0,0,0,0,0]
    else:
        return[best_value, best_move,alpha,beta,value_lst[0],value_lst[1],value_lst[2],value_lst[3],value_lst[4],value_lst[5],value_lst[6]]



@njit(nogil=True, cache=True) 
def num_to_xo(num):
    """
    changes a players number to their symbol
    """
    if num == 0:
        return ' '
    elif num==1:
        return 'X'
    else:
        return 'O'



@njit(nogil=True, cache=True)
def print_board(board):
    """
    prints a nice version of the gameboard
    """
    print('\n')
    print('|| 1 || 2 || 3 || 4 || 5 || 6 || 7 ||')
    print('++---++---++---++---++---++---++---++')
    for n in range(6):
        row = ['| '+num_to_xo(board[k][n])+' |' for k in range(7)]
        row_str = ''
        for s in row:
            row_str+=s
        print('|'+row_str+'|')
        print('++---++---++---++---++---++---++---++')
    print('|| 1 || 2 || 3 || 4 || 5 || 6 || 7 ||')


def print_data(data):
    """
    prints the game board and the score the algorithm gave to each move.
    """
    board = data[:,:6]
    scores = list(data[:,6])
    print('\n')
    print('|| 1 || 2 || 3 || 4 || 5 || 6 || 7 ||')
    print('++---++---++---++---++---++---++---++')
    print('++---++---++---++---++---++---++---++')
    for n in range(6):
        row = ['| '+num_to_xo(board[k][n])+' |' for k in range(7)]
        row_str = ''
        for s in row:
            row_str+=s
        print('|'+row_str+'|')
        print('++---++---++---++---++---++---++---++')
    print('++---++---++---++---++---++---++---++')
    print('|| 1 || 2 || 3 || 4 || 5 || 6 || 7 ||')
    print('++---++---++---++---++---++---++---++')
    print('scores')
    print('||1: {0} ||2: {1} ||3: {2} ||4: {3} ||5: {4} ||6: {5} ||7: {6} ||'.format(*scores))


def print_data_lst(data_lst):
    """
    used to print full games in a list.
    """
    count=0
    for data in data_lst:
        count+=1
        if count%2==0:
            print('\n---------------------------------')
            print('\n\nO\'s turn:' )
        else:
            print('\n---------------------------------')
            print('\n\nX\'s turn:')
        print_data(data)


#The remaining functions are used for data generation.

@njit(nogil=True, cache=True)
def two_ai_game_with_oview(eval_lst,# a list with all of the multipliers
                           x_lvl, # the depth the x-player will look ahead
                           o_lvl, # the depth the o player will look ahead
                           prune, #set to True to use alpha beta pruning
                           len_4_vals, #precomputed values
                           len_5_vals, #precomputed values
                           len_6_vals, #precomputed values
                           len_7_vals): #precomputed values
    """
    Makes two minimaxes play each other, returns a list with all of the game boards, 
    and the scores that the minimax algorithms gave to each move.
    """
    game_board = new_board()
    xrow_mult=eval_lst[0]
    xcol_mult=eval_lst[1]
    xdiag_mult=eval_lst[2]
    orow_mult=eval_lst[3]
    ocol_mult=eval_lst[4]
    odiag_mult=eval_lst[5]
    info_lst=[]
    while not is_game_over(game_board):
        #The "X" player finds their best move.
        data = np.zeros((7,8))
        data[:,:6]=game_board
        mini_lst = minimax(game_board, 
                           True, 
                           x_lvl,
                           x_lvl,
                           xrow_mult,
                           xcol_mult,
                           xdiag_mult, 
                           prune,
                           -1000, 
                           1000,
                           len_4_vals,
                           len_5_vals,
                           len_6_vals,
                           len_7_vals)
        scores = np.array(mini_lst[4:])
        data[:,6] = scores
        info_lst.append(data)
        move = np.array([mini_lst[1]]).astype(np.int32)
        move = move[0]
        if available_moves(game_board)[move-1]==0:
            break
        make_move(game_board, move, 1)
        if available_moves(game_board).max() == 0:
            break
        if not is_game_over(game_board):
            mini_lst = minimax(game_board,
                               False,
                               o_lvl,
                               o_lvl,
                               orow_mult,
                               ocol_mult,
                               odiag_mult,
                               prune,
                               -1000,
                               1000,
                               len_4_vals,
                               len_5_vals,
                               len_6_vals,
                               len_7_vals)
            data = np.zeros((7,8))
            data[:,:6]=game_board
            scores = np.array(mini_lst[4:])
            data[:,6] = scores
            info_lst.append(data)
            move = np.array([mini_lst[1]]).astype(np.int32)
            move = move[0]
            make_move(game_board, move, 2)
    if has_won(game_board, 1):
        for item in info_lst:
            item[:,7] +=1
        return(info_lst)
    elif has_won(game_board, 2):
        for item in info_lst:
            item[:,7] -=1
        return(info_lst)
    else:
        return(info_lst)
    # # print('in two12')

# changed from previous version to only return x-data for both players.  Takes longer since 3 minimax
# subtracting one from 2nd x mini,
# idea is that this is the score x assumed o had when making its move.
# old two_ai_game is now two_ai_game_with_oview
@njit(nogil=True, cache=True)
def two_ai_game(eval_lst,
                x_lvl,
                o_lvl,
                prune,
                len_4_vals,
                len_5_vals,
                len_6_vals,
                len_7_vals):
    """
    Two minimax play each other, returns a list of game boards and the score that the X-player gave to each board.
    """
    game_board = new_board()
    xrow_mult=eval_lst[0]
    xcol_mult=eval_lst[1]
    xdiag_mult=eval_lst[2]
    orow_mult=eval_lst[3]
    ocol_mult=eval_lst[4]
    odiag_mult=eval_lst[5]
    info_lst=[]
    while not is_game_over(game_board):
        #The "X" player finds their best move.
        data = np.zeros((7,8))
        data[:,:6]=game_board
        mini_lst = minimax(game_board, 
                           True, 
                           x_lvl,
                           x_lvl,
                           xrow_mult,
                           xcol_mult,
                           xdiag_mult, 
                           prune,
                           -1000, 
                           1000,
                           len_4_vals,
                           len_5_vals,
                           len_6_vals,
                           len_7_vals)
        scores = np.array(mini_lst[4:])
        data[:,6] = scores
        info_lst.append(data)
        move = np.array([mini_lst[1]]).astype(np.int32)
        move = move[0]
        if available_moves(game_board)[move-1]==0:
            break
        make_move(game_board, move, 1)
        if available_moves(game_board).max() == 0:
            break
        if not is_game_over(game_board):
            mini_lst = minimax(game_board,
                               False,
                               o_lvl,
                               o_lvl,
                               orow_mult,
                               ocol_mult,
                               odiag_mult,
                               prune,
                               -1000,
                               1000,
                               len_4_vals,
                               len_5_vals,
                               len_6_vals,
                               len_7_vals)
            mini_lst_x = minimax(game_board,
                               False,
                               x_lvl-1,
                               x_lvl-1,
                               xrow_mult,
                               xcol_mult,
                               xdiag_mult,
                               prune,
                               -1000,
                               1000,
                               len_4_vals,
                               len_5_vals,
                               len_6_vals,
                               len_7_vals)
            data = np.zeros((7,8))
            data[:,:6]=game_board
            scores = np.array(mini_lst_x[4:])
            data[:,6] = scores
            info_lst.append(data)
            move = np.array([mini_lst[1]]).astype(np.int32)
            move = move[0]
            make_move(game_board, move, 2)
    if has_won(game_board, 1):
        for item in info_lst:
            item[:,7] +=1
        return(info_lst)
    elif has_won(game_board, 2):
        for item in info_lst:
            item[:,7] -=1
        return(info_lst)
    else:
        return(info_lst)




@njit(nogil=True,cache=True)
def two_ai_game_fixed_lvl(xrow,
                          xcol,
                          xdiag,
                          orow,
                          ocol,
                          odiag,
                          prune,
                          len_4_vals,
                          len_5_vals,
                          len_6_vals,
                          len_7_vals):
    """
    Makes two minimaxes play each other, each looking the same depth ahead, 
    returns a list with all of the game boards, 
    and the scores that the minimax algorithms gave to each move.
    """
    x_lvl=3
    o_lvl=3
    game_board = new_board()
    xrow_mult=xrow
    xcol_mult=xcol
    xdiag_mult=xdiag
    orow_mult=orow
    ocol_mult=ocol
    odiag_mult=odiag
    info_lst=[]
    while not is_game_over(game_board):
        data = np.zeros((7,8))
        data[:,:6]=game_board
        mini_lst = minimax(game_board, 
                           True, 
                           x_lvl,
                           x_lvl,
                           xrow_mult,
                           xcol_mult,
                           xdiag_mult, 
                           True,
                           -1000, 
                           1000,
                           len_4_vals,
                           len_5_vals,
                           len_6_vals,
                           len_7_vals)
        scores = np.array(mini_lst[4:])
        data[:,6] = scores
        info_lst.append(data)
        move = np.array([mini_lst[1]]).astype(np.int32)
        move = move[0]
        if available_moves(game_board)[move-1]==0:
            break
        make_move(game_board, move, 1)
        if available_moves(game_board).max() == 0:
            break
        if not is_game_over(game_board):
            mini_lst = minimax(game_board,
                               False,
                               o_lvl,
                               o_lvl,
                               orow_mult,
                               ocol_mult,
                               odiag_mult,
                               True,
                               -1000,
                               1000,
                               len_4_vals,
                               len_5_vals,
                               len_6_vals,
                               len_7_vals)
            data = np.zeros((7,8))
            data[:,:6]=game_board
            scores = np.array(mini_lst[4:])
            data[:,6] = scores
            info_lst.append(data)
            move = np.array([mini_lst[1]]).astype(np.int32)
            move = move[0]
            make_move(game_board, move, 2)
        if available_moves(game_board).max() == 0: 
            break
        if is_game_over(game_board):
            break
    if has_won(game_board, 1):
        for item in info_lst:
            item[:,7] +=1
        return(info_lst)
    elif has_won(game_board, 2):
        for item in info_lst:
            item[:,7] -=1
        return(info_lst)
    else:
        return(info_lst)


@njit(nogil=True,cache=True,parallel=True)
def many_games(N,depth, plus,len_4_vals,len_5_vals,len_6_vals,len_7_vals):
    #plays a lot of games
    eval_lst = [(1.,1.,1.,1.,1.,1.)]
    lst=[]
    recip = 1/N
    for n1 in range(N+plus):
        for n2 in range(N+plus):
            for n3 in range(N+plus):
                for n4 in range(N+plus):
                    for n5 in range(N+plus):
                        for n6 in range(N+plus):
                            if n1==0 and n2==0 and n3==0 and n4==0 and n5 == 0 and n6 ==0:
                                continue
                            else:
                                eval_lst.append((1.+n1*recip,
                                                 1.+n2*recip,
                                                 1.+n3*recip,
                                                 1.+n4*recip,
                                                 1.+n5*recip,
                                                 1.+n6*recip))
    for n in prange(len(eval_lst)):
        lst.append(two_ai_game(eval_lst[n],
                               depth,
                               depth,
                               True,
                               len_4_vals,
                               len_5_vals,
                               len_6_vals,
                               len_7_vals))
    return lst

@njit(nogil=True,cache=True,parallel=True)
def many_games_fixed_x(N,depth_x,depth_o, plus,len_4_vals,len_5_vals,len_6_vals,len_7_vals):
    #plays a lot of games
    eval_lst = [(1.5,1.,1.5,1.,1.,1.)]
    lst=[]
    recip = 1/N
    for n1 in range(N+plus):
        for n2 in range(N+plus):
            for n3 in range(N+plus):
                if n1==0 and n2==0 and n3==0:
                    continue
                else:
                    eval_lst.append((1.5,
                                     1.,
                                     1.5,
                                     1.+n1*recip,
                                     1.+n2*recip,
                                     1.+n3*recip))
    for n in prange(len(eval_lst)):
        lst.append(two_ai_game(eval_lst[n],
                               depth_x,
                               depth_o,
                               True,
                               len_4_vals,
                               len_5_vals,
                               len_6_vals,
                               len_7_vals))
    return lst



@njit(nogil=True,cache=True,parallel=True)
def many_games_var_depth(N,depth, plus=1):
    #plays a lot of games
    eval_lst = [(1.,1.,1.,1.,1.,1.)]
    lst=[]
    recip = 1/N
    len_4_vals = make_n_array(4)
    len_4_vals.flags.writeable = False
    len_5_vals = make_n_array(5)
    len_5_vals.flags.writeable = False
    len_6_vals = make_n_array(6)
    len_6_vals.flags.writeable = False
    len_7_vals = make_n_array(7)
    len_7_vals.flags.writeable = False
    for n1 in range(N+plus):
        for n2 in range(N+plus):
            for n3 in range(N+plus):
                for n4 in range(N+plus):
                    for n5 in range(N+plus):
                        for n6 in range(N+plus):
                            if n1==0 and n2==0 and n3==0 and n4==0 and n5 == 0 and n6 ==0:
                                continue
                            else:
                                eval_lst.append((1.+n1*recip,
                                                 1.+n2*recip,
                                                 1.+n3*recip,
                                                 1.+n4*recip,
                                                 1.+n5*recip,
                                                 1.+n6*recip))
    for n in prange(len(eval_lst)):
        lst.append(two_ai_game(eval_lst[n],depth,depth))
        lst.append(two_ai_game(eval_lst[n],depth-1,depth))
        lst.append(two_ai_game(eval_lst[n],depth,depth-1))
    return lst




@njit(nogil=True, cache=True)
def into_mini(board,player,result):
    #used to change move scores
    if player == 1:
        is_max=True
    else:
        is_max = False
    array = minimax(board,
                    is_max,
                    3,#depth
                    3,#start_depth,
                    2.0, #row_mult,
                    1.0, #col_mult,
                    2.0, #diag_mult 
                    -1000, #alpha 
                    1000) #beta
    to_return = np.zeros((7,8))
    to_return[:,:6]=board
    to_return[:,7]=array[4:]
    to_return[:,6]=np.ones(7)*result
    return to_return



@njit( cache=True)
def make_score_array(states,players,results):
    lst=[]
    for n in prange(len(states)):
        lst.append(into_mini(states[n],players[n],results[n]))
    return(lst)


def board_to_ar_nb(board):
    num_board = []
    for col in board:
        for spot in col:
            num_board.append(spot)
    return(np.array(num_board))


def result_to_ar_nb(result):
    if result == 1:
        return(np.array([1,0,0]))
    elif result == 0:
        return(np.array([0,1,0]))
    else:
        return(np.array([0,0,1]))

def results_cleaned(results_lst):
    boards = np.zeros((len(results_lst),42))
    results = np.zeros((len(results_lst),3))
    scores = np.zeros((len(results_lst),7))
    for n in range(len(results_lst)):
        # print(n)
        result = results_lst[n]
        boards[n] = board_to_ar_nb(result[:,:6])
        results[n] = result_to_ar_nb(result[1,7])
        scores[n] = result[:,6]
    return([boards.T,scores.T,results.T])


#import pickle
# def open_file(file='ar.pkl'):
#     with open(file,'rb+') as pkl_file:
#         ar=pickle.load(pkl_file)
#         pkl_file.close()
#     return(ar)


# def save_file(clean_results,file='brs.pkl'):
#     with open(file,'wb+') as pkl_file:
#         pickle.dump(clean_results,pkl_file,-1)
#         #print('saved')
#         pkl_file.close()


