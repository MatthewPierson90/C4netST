# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:41:44 2021

@author: Matthew
"""

from CNNoperations import ssigmoid
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle
############################################################
def rlen(lst,start =0):
    return range(start,len(lst))


def sigmoid(z):
    return(1/(1+np.exp(-z)))

def shuffle_all_data(data):
    length = data[0].shape[1]
    shuff = [n for n in range(length)]
    np.random.shuffle(shuff)
    data0 = data[0][:,shuff]
    data1 = data[1][:,shuff]
    data2 = data[2][:,shuff]
    return((data0,data1,data2))


def shuffle_all_data_unflat(data):
    length = len(data[0])
    shuff = [n for n in range(length)]
    np.random.shuffle(shuff)
    data0 = np.array([data[0][n] for n in shuff])
    data1 = np.array([data[1][n] for n in shuff])
    data2 = np.array([data[2][n] for n in shuff])
    # data2[data2==-1]=2
    return([data0,data1,data2])



def shuffle_mcts_data(data):
    length = len(data[0])
    shuff = [n for n in range(length)]
    np.random.shuffle(shuff)
    data0 = np.array([data[0][n] for n in shuff])
    data1 = np.array([data[1][n] for n in shuff])
    data2 = np.array([data[2][n] for n in shuff])
    # data2[data2==-1]=2
    return((data0,data1,data2))

def shuffle_training_data(data):
    length = data[0].shape[1]
    shuff = [n for n in range(length)]
    np.random.shuffle(shuff)
    data0 = data[0][:,shuff]
    data1 = data[1][:,shuff]
    return((data0,data1))

def board_to_array(board):
    num_board = []
    for col in board:
        for spot in col:
            if spot == ' ':
                num_board.append(0)
            elif spot == 'X':
                num_board.append(1)
            else:
                num_board.append(2)
    return(np.array(num_board).reshape(len(num_board),1))

def array_to_move(vect):
    return(vect.argmax()+1)


def move_to_array(move):
    lst = [0 for n in range(7)]
    lst[move-1]=1
    return(np.array(lst).reshape(len(lst),1))

def array_to_result(vect):
    arg_max = vect.argmax()
    if arg_max == 0 :
        return 1
    elif arg_max ==1:
        return 0
    else:
        return -1

def num_to_result(num):
    if num == 0:
        return 1
    if num == 1:
        return 0
    if num==2:
        return -1

def result_to_array(result):
    if result == 1:
        return(np.array([[1],[0],[0]]))
    elif result == 0:
        return(np.array([[0],[1],[0]]))
    else:
        return(np.array([[0],[0],[1]]))

def array_to_board(array,h=7,w=6):
    if array.ndim == 1:
        board = array.reshape(h,w)
    else:
        transpose_array = array.T
        num = transpose_array.shape[0]
        board = transpose_array.reshape(num,h,w)
    return(board)

# if __name__=='__main__':
#     check = np.array([n for n in range(42)])
#     check3 = np.array([[n for n in range(42)]]*1000000).T
#     check2 = array_to_board(check)
#     print(check2)
#     print(array_to_board(check3).shape)
    
def save_data(data,file='base_layer.pkl'):
    #----------------------------------------------------------
    with open(file,'wb+') as pkl_file:
        pickle.dump(data,pkl_file,-1)
        #print('saved')
        pkl_file.close()
#----------------------------------------------------------


def open_file(file='ar.pkl'):
    #----------------------------------------------------------
    with open(file,'rb+') as pkl_file:
        ar=pickle.load(pkl_file)
        pkl_file.close()
    return(ar)

def get_test_data():
    test_data = open_file('CNN_test_data.pkl')
    test_inputs = test_data[0]
    test_y_values = test_data[2]
    return((test_inputs,test_y_values))

def get_score_data():
    big_set = open_file('data_with_scores_depth6.pkl')
    small_set = open_file('data_with_scores2.pkl')
    size_1 = big_set[0].shape[1]
    size_2 = small_set[0].shape[1]
    tsize = size_1+size_2
    states = np.zeros((42,tsize))
    scores = np.zeros((7,tsize))
    results = np.zeros((3,tsize))
    states[:,:size_1] = big_set[0]
    states[:,size_1:] = small_set[0]
    scores[:,:size_1] = big_set[2]
    scores[:,size_1:] = small_set[2]
    results[:,:size_1] = big_set[1]
    results[:,size_1:] = small_set[1]
    return [states,scores,results]



def results_cleaned(results_lst):
    boards = []
    scores = []
    results = []
    for lst in results_lst:
        for n in range(len(lst)):
            result = lst[n]
            boards.append(result[:,:6])
            results.append(result[1,7])
            scores.append(result[:,6])
    return([boards,scores,results])


def make_data_for_CNN_flattened():
    data = open_file('data_with_scores.pkl')
    data = shuffle_all_data(data)
    board_lst = []
    result_lst = []
    for n in rlen(data[0].T):
        item1 =data[0].T[n]
        item2 = data[2].T[n]
        board_lst.append(array_to_board(item1))
        result_lst.append(item2.T.argmax())
    new_data = [np.array(board_lst),data[1],np.array(result_lst)]
    return(new_data)


def make_data_for_CNN_unflattened():
    data = open_file('old_data/data_w_scores_x_view1.pkl')
    data = results_cleaned(data)
    data = shuffle_all_data_unflat(data)
    boards1 = deepcopy(data[0])
    boards1copy1 = deepcopy(boards1)
    boards1copy1[boards1copy1==2]=-1
    boards_lst = []
    for n in rlen(boards1):
        board1 = boards1[n]
        board2 = boards1copy1[n]
        board1[board1==2]=0
        board2[board2==1]=0
        two_boards = np.array([board1,board2])
        boards_lst.append(two_boards)
    boards2=np.array(boards_lst)
    boards = np.moveaxis(boards2, 1, -1)
    data[0] = boards
    turn_count_lst = []
    for boards in data[0]:
        x_moves = boards[:,:,0].sum()
        o_moves = boards[:,:,1].sum()
        turn_count = x_moves-o_moves+1
        turn_count_lst.append(turn_count)
    data.insert(0,turn_count_lst )
    fixed_results = []
    for n in rlen(data[0]):
        turn_count = data[0][n]
        result = data[-1][n]
        
        if turn_count%2==1 and result==1:
            fixed_results.append(1.)
        elif turn_count%2==0 and result==-1:
            fixed_results.append(1.)
        else:
            fixed_results.append(0.)
    fixed_scores = []
    for n in rlen(data[0]):
        turn_count = data[0][n]
        scores = data[-2][n]
        
        if turn_count%2==1:
            fixed_scores.append(scores)
        elif turn_count%2==0:
            fixed_scores.append(-scores)
    data.append(np.array(fixed_scores))
    data.append(np.array(fixed_results))
    return(data)

# data = make_data_for_CNN_unflattened()
# init_boards = np.array(data[1])
# init_scores = data[4]
# init_results = data[5]

def train_test_split(data,test_size=None):
    data = shuffle_all_data(data)
    if test_size == None:
        test_size = round(.1*data[0].shape[1])
    test_boards = data[0][:,0:test_size]
    test_moves = data[1][:,0:test_size]
    test_results = data[2][:,0:test_size]
    train_boards = data[0][:,test_size:]
    train_moves = data[1][:,test_size:]
    train_results = data[2][:,test_size:]
    test_data = [test_boards,test_moves,test_results]
    train_data = [train_boards,train_moves,train_results]
    
    return(train_data,test_data)


# def fix_first_mcts_data():
#     data = open_file('Data_MCTS_W_C4NET_It1.pkl')
#     for item in data:
#         boards = item[0]
#         x_moves = boards[:,:,0].sum()
#         o_moves = boards[:,:,1].sum()
#         turn_count = x_moves-o_moves+1
#         item.insert(0,turn_count)
#     save_data(data,'Data_MCTS_W_C4NET_It1_added_turn_count.pkl')

def mcts_data(file):
    data = open_file(file)
    fixed_results = []
    for item in data:
        if item[0]%2==1 and item[-1]==1:
            fixed_results.append(1)
        elif item[0]%2==0 and item[-1]==2:
            fixed_results.append(1)
        else:
            fixed_results.append(0)
    boards = np.array([item[1] for item in data])
    scores = np.array([item[2] for item in data])
    results = np.array(fixed_results)
    return(boards,scores,results)


#  
# data = get_score_data()
# data = make_data_for_CNN_unflattened()
# # data = make_data_for_CNN_flattened()
# boards1 = deepcopy(data[0])
# scores = data[1]
# results = data[2]
# boards1copy1 = deepcopy(boards1)
# boards1copy1[boards1copy1==2]=-1
# boards_lst = []
# for n in rlen(boards1):
#     board1 = boards1[n]
#     board2 = boards1copy1[n]
#     board1[board1==2]=0
#     board2[board2==1]=0
#     two_boards = np.array([board1,board2])
#     boards_lst.append(two_boards)
# boards2=np.array(boards_lst)
# boards = np.moveaxis(boards2, 1, -1)