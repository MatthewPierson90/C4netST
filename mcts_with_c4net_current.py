# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:02:46 2021

@author: Matthew
"""
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(50)
import random
import os
import pickle
import numpy as np
from copy import deepcopy
import time
tt = time.time
# from numba_board_functions import available_moves,has_won,make_move,new_board
from Using_numba_current import has_won,new_board,minimax
rng = np.random.default_rng()
np.set_printoptions(suppress=True)
# set_tf_loglevel(100)
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


def make_eval_function(to_load='C:/Users/matth/python/connectfour/c4netST/Version 2/c4netST_versions/c4netST_0_0'):
    import os
    import logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(50)
    from tensorflow import function, TensorSpec
    from tensorflow import lite, device
    from tensorflow.keras.models import load_model
    class lite_model_class:
        # Majority of code written by Michael Wurm
        # https://micwurm.medium.com/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98
        # I added code for second output, and for concrete_function class method, 
        # removed predict for more than one input.
        
        @classmethod
        def from_file(cls, model_path):
            return lite_model_class(lite.Interpreter(model_path=model_path))
        
        @classmethod
        def from_keras_model(cls, kmodel):
            converter = lite.TFLiteConverter.from_keras_model(kmodel)
            tflite_model = converter.convert()
            return lite_model_class(lite.Interpreter(model_content=tflite_model))
        
        @classmethod
        def from_concrete_function(cls,func):
            converter = lite.TFLiteConverter.from_concrete_functions([func])
            tflite_model = converter.convert()
            return lite_model_class(lite.Interpreter(model_content=tflite_model))
        
        def __init__(self, interpreter):
            self.interpreter = interpreter
            self.interpreter.allocate_tensors()
            input_det = self.interpreter.get_input_details()[0]
            output0_det = self.interpreter.get_output_details()[0]
            output1_det = self.interpreter.get_output_details()[1]
            self.input_index = input_det["index"]
            self.input_shape = input_det["shape"]
            self.input_dtype = input_det["dtype"]
            
            self.output0_index = output0_det["index"]
            self.output0_shape = output0_det["shape"]
            self.output0_dtype = output0_det["dtype"]
            
            self.output1_index = output1_det["index"]
            self.output1_shape = output1_det["shape"]
            self.output1_dtype = output1_det["dtype"]
            
        
        def predict(self, inp):
            """ Like predict(), but only for a single record. The input data can be a Python list. """
            inp = np.array([inp], dtype=self.input_dtype)
            self.interpreter.set_tensor(self.input_index, inp)
            self.interpreter.invoke()
            out0 = self.interpreter.get_tensor(self.output0_index)
            out1 = self.interpreter.get_tensor(self.output1_index)
            return [out0[0],out1[0]]
    
    with device('/CPU:0'):
        model = load_model(to_load)
        model_eval = function(model,input_signature = [TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)])
        concrete_model_eval = model_eval.get_concrete_function()
        lite_model = lite_model_class.from_concrete_function(concrete_model_eval)
    # lite_eval = lite_model.predict
    # set_tf_loglevel(0)
    # del lite
    del load_model
    del model
    del function
    del TensorSpec
    del model_eval
    del concrete_model_eval
    return(lite_model)


def open_data(file='base_layer.pkl'):
    #----------------------------------------------------------
    with open(file,'rb+') as pkl_file:
        layer1=pickle.load(pkl_file)
        pkl_file.close()
    return(layer1)
#----------------------------------------------------------

def save_data(data,file='base_layer.pkl'):
    #----------------------------------------------------------
    with open(file,'wb+') as pkl_file:
        pickle.dump(data,pkl_file,-1)
        #print('saved')
        pkl_file.close()
#----------------------------------------------------------


def rlen(lst,start =0):
    return range(start,len(lst))


def board_str(board):
    #-------------------------------------------------------------------
    bs ='\n|1|2|3|4|5|6|7|\n+-+-+-+-+-+-+-+\n'
    for n in range(6):
        bs+='|'
        for m in range(7):
            if board[m][n] == 0:
                to_add = ' '
            elif board[m][n] == 1:
                to_add = 'X'
            else:
                to_add = 'O'
            bs+=to_add+'|'
        bs+='\n'
    return(bs)
    #-------------------------------------------------------------------


def board_str_save(board):
    #-------------------------------------------------------------------
    bs =''
    for n in range(6):
        for m in range(7):
            if board[m,n]==' ':
                bs+='S'
            else:
                bs+=board[m,n]
    return(bs)
    #-------------------------------------------------------------------


def available_moves(board):
    moves = np.zeros(7)
    for n in range(7):
        if board[n][0]==0:
            moves[n]=1
    return(moves.astype(np.int32))



def make_move(board, col, player):
    # print('in make move')
    for n in range(6):
        if n == 5:
            board[col-1][n] = player
            break
        elif board[col-1][n+1]!= 0:
            board[col-1][n] = player
            break
    return(board)


class node_class(object):
    def __init__(self,
                 state,
                 model_eval_func,
                 dirc = None,
                 player='X',
                 player_num = 1,
                 add_noise = True,
                 true_root = None):
        # toc=tt()
        if true_root == None:
            self.true_root = self
            self.edges_visited = set([])
            dirc = rng.dirichlet([1,2,3,4,5,6,7])
        else:
            self.true_root = true_root

        self.dirc = dirc
        self.add_noise = add_noise
        self.state = deepcopy(state)
        self.x_state = deepcopy(state)
        self.o_state = deepcopy(state)
        self.x_state[self.x_state==2]=0
        self.o_state[self.o_state==1]=0
        self.o_state[self.o_state==2]=-1
        self.split_board = np.array([self.x_state,self.o_state])
        self.split_board = np.moveaxis(self.split_board, 0, -1)
        self.boardstr=board_str(self.state)
        # tic = tt()
        
        # print('make info board time:',tic-toc)
        # toc = tt()
        self.moves = available_moves(state)
        # tic = tt()
        # print('finding move time:',tic-toc)
        self.player = player
        self.player_num = player_num
        self.model_eval_func = model_eval_func
        self.is_leaf = False
        if has_won(state,1):
            self.is_leaf = True
            self.result =1
        elif has_won(state,2):
            self.is_leaf = True
            self.result =2
        elif max(self.moves)==0:
            self.is_leaf = True
            self.result = 0
        if not self.is_leaf:
            # toc = tt()
            # self.model_P = self.model_eval_func(np.array([self.split_board]))[0].numpy()[0]
            model_P,model_V = self.model_eval_func.predict(self.split_board)
            self.model_V = model_V[0]
            self.model_P = model_P
            # tic=tt()
            # print('evaluate model time:',tic-toc)
            # toc=tt()
            self.edges = [edge_class(state=state,
                                     action = move+1,
                                     model_eval_func = model_eval_func,
                                     model_P = self.model_P,
                                     dirc = self.dirc,
                                     source_node = self,
                                     player = self.player,
                                     player_num = self.player_num,
                                     add_noise = self.add_noise,
                                     true_root = self.true_root) for move in range(7) if self.moves[move]==1]
            self.pi = np.zeros(7)
            self.layer_N = 0
            # tic = tt()
            # print('makeing edge time:',tic-toc)

            self.update_edge_with_max_av()
            
    def update_pi(self):
        self.max_pi = 0
        for edge in self.edges:
            new_pi = edge.N/self.layer_N
            self.pi[edge.action-1] = new_pi
            if self.max_pi<new_pi:
                self.max_pi = new_pi
                self.edge_max_pi = edge
    def update_edge_with_max_av(self):
            self.max_av=-np.inf
            for edge in self.edges:
                if edge.av > self.max_av:
                    self.max_av = edge.av
                    self.edge_with_max_av = edge
    def reset(self,new_dirc = None):
        self.pi = np.zeros(7)
        self.layer_N = 0
        self.max_av=-np.inf
        self.max_pi = 0
        # if self.add_noise and (new_dirc!=None)[0]:
        #     self.dirc = new_dirc
        #     for edge in self.edges:
        #         edge.P = .75*self.model_P[edge.action-1]+.25*self.dirc[edge.action-1]
    # def __eq__(self,other):
    #     #-------------------------------------------------------------------
    #     if not isinstance(other, node_class):
    #         return NotImplemented
    #     return(self.state == other.state and self.action == other.action)
    # #-------------------------------------------------------------------
    
    def __str__(self):
        #-------------------------------------------------------------------
        return 'Board:\n{0} \nPlayer: {1}\n'.format(self.boardstr,self.player)
    #-------------------------------------------------------------------
        
    def __repr__(self):
        #-------------------------------------------------------------------
        return str(self)
    #-------------------------------------------------------------------


class edge_class(object):
    ####################################################################
    def __init__(self,
                 state,
                 action,
                 model_eval_func,
                 model_P,
                 dirc,
                 source_node,
                 player = 'X',
                 player_num = 1,
                 add_noise = True,
                 true_root = None):
        #-------------------------------------------------------------------
        self.true_root = true_root
        self.state =state
        self.boardstr=board_str(self.state)
        self.action = action
        self.model_eval_func = model_eval_func
        self.player =player
        self.player_num = player_num
        self.source_node = source_node
        
        self.made_target_node = 0
        self.add_noise = add_noise
        self.model_P = model_P
        self.dirc = dirc
        if add_noise:
            self.P = .75*model_P[self.action-1]+.25*dirc[self.action-1]
        else:
            self.P = model_P[self.action-1]
        self.model_source_V=self.source_node.model_V
        self.N = 0 #number of visits
        self.Q = 0 #score
        self.C = 1 #constant
        self.W=0
        self.Pi=0
        preU=self.C*self.P
        self.U = preU #base ratio
        self.av = self.Q+self.U #action value
        #-------------------------------------------------------------------
    def make_target_node(self):
        if self.made_target_node == 1:
            return self.target_node
        else:
            state_copy = deepcopy(self.state)
            new_state = make_move(state_copy,self.action,self.player_num)
            if self.player_num ==1:
                self.target_node = node_class(state = new_state,
                                              model_eval_func = self.model_eval_func,
                                              dirc = self.dirc,
                                              player = 'O',
                                              player_num = 2,
                                              add_noise = self.add_noise,
                                              true_root = self.true_root)
            else:
                self.target_node = node_class(state = new_state,
                                              model_eval_func = self.model_eval_func,
                                              dirc = self.dirc,
                                              player = 'X',
                                              player_num = 1,
                                              add_noise = self.add_noise,
                                              true_root = self.true_root)
            self.made_target_node =1
            if not self.target_node.is_leaf:
                self.model_source_V_WRT_target=1-self.target_node.model_V #source's win percent with respect to target's view
            else:
                self.model_source_V_WRT_target=1
            self.V_diff = self.model_source_V_WRT_target-self.model_source_V
            return self.target_node
    def update(self,result):
        #-------------------------------------------------------------------
        self.N+=1
        self.source_node.layer_N+=1
        if result == self.player_num:
            self.W+=1
        else:
            self.W -=1
        self.C1=1
        self.C2=0
        preU=self.C1*self.P*np.sqrt(self.source_node.layer_N) /(1+self.N)+self.C2*self.V_diff
        self.U = preU
        self.Q = self.W/(1+self.N)
        self.av = self.Q+self.U
    def update_with_v(self,w_guess,new_player_num):
        #-------------------------------------------------------------------
        self.N+=1
        self.source_node.layer_N+=1
        if new_player_num == self.player_num:
            self.W += w_guess
        else:
            self.W -= w_guess
        self.C1=1
        self.C2=0
        preU=self.C1*self.P*np.sqrt(self.source_node.layer_N) /(1+self.N)+self.C2*self.V_diff
        self.U = preU
        self.Q = self.W/(1+self.N)
        self.av = self.Q+self.U
    #-------------------------------------------------------------------
    def reset(self,reset_dirc = False,new_dirc = None):
        self.made_target_node = 2
        self.source_node.reset()
        if reset_dirc and self.add_noise:
            self.dirc = new_dirc
            self.P = .75*self.model_P[self.action-1]+.25*new_dirc[self.action-1]
        self.N = 0 #number of visits
        self.Q = 0 #score
        self.C = 1 #constant
        self.W=0
        self.Pi=0
        preU=self.C*self.P
        self.U = preU #base ratio
        self.av = self.Q+self.U

    
    # def __eq__(self,other):
    #     #-------------------------------------------------------------------
    #     if not isinstance(other, edge_class):
    #         return NotImplemented
    #     return(self.state == other.state and self.action == other.action)
    # #-------------------------------------------------------------------
    
    def __str__(self):
        #-------------------------------------------------------------------
        return 'board:\n{0}making move:{1} \nPlayer:{2}\n'.format(self.boardstr,self.action,self.player)
    #-------------------------------------------------------------------
        
    def __repr__(self):
        #-------------------------------------------------------------------
        return str(self)
    #-------------------------------------------------------------------
        
        
####################################################################


def tree_node_edge_count(root_node):
    node_count = 1
    edge_count =0
    if not root_node.is_leaf:
        for edge in root_node.edges:
            edge_count+=1
            if edge.made_target_node==1:
                to_add_node_count,to_add_edge_count=tree_node_edge_count(edge.target_node)
                node_count+=to_add_node_count
                edge_count+=to_add_edge_count
        return(node_count,edge_count)
    else:
        return(1,0)


def print_node_edge_info(node,print_all = False,av=True,W=False,N=False,layer_N=False, P=False,U=False,Q=False):
    if print_all:
        print('Node visit count:',node.layer_N)
        count = 0
        for edge in node.edges:
            print('\n--------------------------------\n')
            print('Edge List Index:',count)
            count+=1
            print('Edge Action:',edge.action)
            print('Edge Taken Count:',edge.N)
            print('Edge Win Difference:',edge.W)
            print('Model Prob Given:',round(edge.P,3))
            print('Current U:',round(edge.U,3))
            print('Current Q:',round(edge.Q,3))
            print('Current av:',round(edge.av,3))
    else:
        if layer_N:
            print('Node visit count:',node.layer_N)
        count = 0
        for edge in node.edges:
            print('\n--------------------------------\n')
            print('Edge Action:',edge.action)
            print('Edge List Index:',count)
            count+=1
            if N:
                print('Edge Taken Count:',edge.N)
            if W:
                print('Edge Win Difference:',edge.W)
            if P:
                print('Model Prob Given:',round(edge.P,3))
            if U:
                print('Current U:',round(edge.U,3))
            if Q:
                print('Current Q:',round(edge.Q,3))
            if av:
                print('Current av:',round(edge.av,3))

def make_new_tree(model,add_noise = True):
    return(node_class(new_board(),model,add_noise=add_noise))


def simulate(root_node,
             itterations=2,
             reset_edges = True):
    num_moves = 0
    for n in range(itterations):
        new_node = root_node
        edge_lst =[]
        made_node = 0
        while not new_node.is_leaf and made_node != 1:
            new_node.update_edge_with_max_av()
            new_edge = new_node.edge_with_max_av
            edge_lst.append(new_edge)
            if reset_edges:
                new_node.true_root.edges_visited.add(new_edge)
            if new_edge.made_target_node != 1:
                made_node = 1
            if new_edge.made_target_node == 2:
                new_edge.made_target_node = 1
            new_node = new_edge.make_target_node()
            num_moves+=1
        if new_node.is_leaf:
            result = new_node.result
            for edge in edge_lst:
                edge.update(result)
        else:
            v = new_node.model_V
            w_guess = 2*v-1
            new_player_num = new_node.player_num
            for edge in edge_lst:
                edge.update_with_v(w_guess,new_player_num)
    root_node.update_pi()
    return(num_moves)


# version_path = 'C:/Users/matth/python/connectfour/c4netST/V2/c4netST_versions/c4netST_1-1'
# model = make_eval_function(version_path)
# root_node = make_new_tree(model)
# toc = tt()
# print(simulate(root_node,1000))
# tic = tt()
# print(root_node.pi)
# print(tic-toc)
# print(len(root_node.true_root.edges_visited))


def self_train(model = None,
               root_node = None,
               itterations = 20,
               print_true = True,
               reset_edges = False,
               print_final = True):
    if root_node == None:
        root_node = make_new_tree(model)
    data = []
    sim_time = 0
    turn_count =1
    while not root_node.is_leaf:
        toc = tt()
        simulate(root_node,itterations, reset_edges)
        tic = tt()
        sim_time += tic-toc
        root_node.update_pi()
        pi = root_node.pi
        data.append([turn_count,np.copy(root_node.split_board),np.copy(pi)])
        turn_count+=1
        player = root_node.player
        move = pi.argmax()
        if print_true:
            print('Current Board:\n')
            print(root_node.boardstr)
            
            statement = player +'s Turn'
            print('\n------------------------------------------------\n')
            print('\n',statement,'\n')
            
            print(player,'chose ',move+1)
            print(pi)
        root_node = root_node.edge_max_pi.make_target_node()
    if print_true:
        print(root_node.boardstr)
        print('Avg sim time:',sim_time/(turn_count-1))
    if print_final and not print_true:
        print('Final move:',player,'chose ',move+1)
        print(root_node.boardstr)
        print('Avg sim time:',sim_time/(turn_count-1))
    for item in data:
        item.append(root_node.result)
    if reset_edges:
        toc = tt()
        print('num edges visited:',len(root_node.true_root.edges_visited))
        for edge in root_node.true_root.edges_visited:
            edge.reset()
        root_node.true_root.edges_visited = set([])
        tic = tt()
        print('Reset time:',tic-toc)
    return(root_node.result,data)

def self_train_test(model = None,
               root_node = None,
               itterations = 20,
               print_true = True,
               reset_edges = False,
               print_final = True):
    if root_node == None:
        root_node = make_new_tree(model)
    data = []
    sim_time = 0
    turn_count =1
    while not root_node.is_leaf:
        toc = tt()
        simulate(root_node,itterations, reset_edges)
        tic = tt()
        sim_time += tic-toc
        root_node.update_pi()
        pi = root_node.pi
        data.append([turn_count,root_node.split_board,pi])
        turn_count+=1
        player = root_node.player
        move = pi.argmax()
        if print_true:
            print('Current Board:\n')
            print(root_node.boardstr)
            
            statement = player +'s Turn'
            print('\n------------------------------------------------\n')
            print('\n',statement,'\n')
            
            print(player,'chose ',move+1)
            print(pi)
        root_node = root_node.edge_max_pi.make_target_node()
    if print_true:
        print(root_node.boardstr)
        print('Avg sim time:',sim_time/(turn_count-1))
    if print_final and not print_true:
        print('Final move:',player,'chose ',move+1)
        print(root_node.boardstr)
        print('Avg sim time:',sim_time/(turn_count-1))
    for item in data:
        item.append(root_node.result)
    if reset_edges:
        
        toc = tt()
        # to_reset = 8
        # count = np.random.randint(0,8)
        print('num edges visited:',len(root_node.true_root.edges_visited))
        new_dirc = rng.dirichlet([1,2,3,4,5,6,7])
        half_edges = len(root_node.true_root.edges_visited)/2
        count = 0
        for edge in root_node.true_root.edges_visited:
            if count>half_edges:
                reset_dirc= True
            else:
                reset_dirc = False
            edge.reset(reset_dirc =reset_dirc,new_dirc = new_dirc)
            count+=1
        root_node.true_root.edges_visited = set([])
        tic = tt()
        print('Reset time:',tic-toc)

    return(sim_time,turn_count-1)

# if __name__ == '__main__':
#     version_path = 'C:/Users/matth/python/connectfour/c4netST/V2/c4netST_versions/c4netST_1-1'
#     model = make_eval_function(version_path)
    

#     time = 0
#     itts = 50
#     total_sim_time = 0
#     total_sim_itts = 0
#     for n in range(itts):
#         if n%25 == 0:
#             root_node = make_new_tree(model = model, add_noise = True)
#         print('\nGame',n+1)
#         toc =tt()
#         sim_time,sim_itts = self_train_test(root_node = root_node,
#                                        itterations =1000,
#                                        print_true = False,
#                                        reset_edges = True,
#                                        print_final = True)
#         tic = tt()
#         total_sim_time+=sim_time
#         total_sim_itts+=sim_itts
#         time += tic-toc
#         print('game time:',tic-toc)
#         print('Current avg time:',(time)/(n+1))
#         print('Total sim time avg:',total_sim_time/total_sim_itts)
#     print('Avg time:',(time)/itts)
#     print('')
#     print('-'*50)
#     print('')
#     del root_node
#     del model
#     import tensorflow.keras.backend as K
#     K.clear_session()
#     model = make_eval_function(version_path)
#     time = 0
#     for n in range(itts):
#         print('\nGame',n+1)
#         root_node = make_new_tree(model = model, add_noise = True)
#         toc =tt()
#         sim_time,sim_itts = self_train_test(root_node = root_node,
#                                        itterations =1000,
#                                        print_true = False,
#                                        reset_edges = False,
#                                        print_final = True)
#         tic = tt()
#         total_sim_time+=sim_time
#         total_sim_itts+=sim_itts
#         time += tic-toc
#         print('game time:',tic-toc)
#         print('Current avg time:',(time)/(n+1))
#         print('Total sim time avg:',total_sim_time/total_sim_itts)
#     print('Avg time:',(time)/itts)





# def soft(array):
#     to_return = np.exp(array)
#     array_sum = to_return.sum()
#     to_return = to_return/array_sum
#     return(to_return)
# from Using_Keras_current import soft

def play_mini(root_node,
              as_x = True,
              itterations = 100,
              print_true = True,
              depth = 4,
              row_mult = 1.5,
              col_mult = 1,
              diag_mult = 1.5,
              len_4_vals = open_data('array_n_values/len_4_values.pkl'),
              len_5_vals = open_data('array_n_values/len_5_values.pkl'),
              len_6_vals = open_data('array_n_values/len_6_values.pkl'),
              len_7_vals = open_data('array_n_values/len_7_values.pkl'),
              print_final_board = False):
    data = []
    turn_count =1
    if as_x:
        is_x=1
    else:
        is_x=0
    while not root_node.is_leaf:
        if turn_count%2==is_x:
            chooser = 'c4netST'
            simulate(root_node,itterations)
            root_node.update_pi()
            pi = root_node.pi
            # print(pi)
            move = pi.argmax()
            move = move+1
            data.append([turn_count,np.copy(root_node.split_board),np.copy(pi)])
        else:
            chooser = 'Mini'
            mini_lst = minimax(root_node.state, 
                           not as_x, 
                           depth,
                           depth,
                           row_mult,
                           col_mult,
                           diag_mult, 
                           prune = True,
                           alpha = -1000, 
                           beta = 1000,
                           len_4_vals = len_4_vals,
                           len_5_vals = len_5_vals,
                           len_6_vals = len_6_vals,
                           len_7_vals = len_7_vals)
            move = mini_lst[1]
        turn_count+=1
        if print_true:
            print('Current Board:\n')
            print(root_node.boardstr)
            player = chooser+' (Player '+root_node.player+')'
            statement = player +'s Turn'
            print('\n------------------------------------------------\n')
            print('')
            print(statement)
            print('')
            print(player,'chose ',move)
        for edge in root_node.edges:
            if edge.action == move:
                root_node = edge.make_target_node()
                break
    if print_true:
        print(root_node.boardstr)
    if print_final_board and not print_true:
        print(chooser,'won!')
        print('Final move:',move)
        print('Final Board:')
        print(root_node.boardstr)
    for item in data:
        item.append(root_node.result)
    beat_mini=0
    if as_x and root_node.result==1:
        beat_mini=1
    elif not as_x and root_node.result==2:
        beat_mini=1
    # del root_node
    return(beat_mini,data)

def multi_play_mini(as_x,
                    len_4_vals,
                    len_5_vals,
                    len_6_vals,
                    len_7_vals,
                    depth,
                    in_game_itterations = 200,
                    training_itterations = 2000,
                    num_games = 25,
                    version_path = 'C:/Users/matth/python/connectfour/c4netST/V2/c4netST_versions/c4netST_0-0'):
    model = make_eval_function(version_path)
    # root_node = make_new_tree(model = model, add_noise = False)
    depths = []
    for n in range(1,depth+1):
        if n == depth:
            depths+=[depth]*(depth)
        elif n==depth-1:
            depths+=[n]*2
        else:
            depths+=[n]
    data_lst = []
    beat_mini = 0
    # simulate(root_node,training_itterations)
    reset = True
    
    for n in range(num_games):
        if n%25 == 0:
            root_node = make_new_tree(model = model, add_noise = True)
        row_mult=random.uniform(1,4)
        col_mult=random.uniform(1,4)
        diag_mult=random.uniform(1,4)
        result,data = play_mini(root_node,
                                as_x = as_x,
                                itterations = in_game_itterations,
                                print_true = False,
                                depth = depths[np.random.randint(0,len(depths))],
                                row_mult = row_mult,
                                col_mult = col_mult,
                                diag_mult = diag_mult,
                                len_4_vals=len_4_vals,
                                len_5_vals=len_5_vals,
                                len_6_vals=len_6_vals,
                                len_7_vals=len_7_vals)
        data_lst +=data
        beat_mini+=result
        new_dirc = rng.dirichlet([1,2,3,4,5,6,7])
        half_edges = len(root_node.true_root.edges_visited)/2
        count = 0
        for edge in root_node.true_root.edges_visited:
            if count>half_edges:
                reset_dirc= reset
            else:
                reset_dirc = not reset
            edge.reset(reset_dirc =reset_dirc,new_dirc = new_dirc)
            count+=1
        root_node.true_root.edges_visited = set([])
        if n%4:
            reset = not reset
        root_node.true_root.edges_visited = set([])
    root_node = make_new_tree(model = model, add_noise = False)
    del root_node
    del model
    import tensorflow.keras.backend as K
    K.clear_session()
    return [beat_mini,data_lst]


def multi_play_mini_test(as_x,
                         len_4_vals,
                         len_5_vals,
                         len_6_vals,
                         len_7_vals,
                         depth,
                         in_game_itterations = 200,
                         training_itterations = 2000,
                         num_games = 25,
                         version_path = 'C:/Users/matth/python/connectfour/c4netST/V2/c4netST_versions/c4netST_0-0'):
    model = make_eval_function(version_path)
    # root_node = make_new_tree(model = model, add_noise = False)
    row_mult=random.uniform(1,4)
    col_mult=random.uniform(1,4)
    diag_mult=random.uniform(1,4)
    depths = []
    for n in range(1,depth+1):
        if n == depth:
            depths+=[depth]*(depth)
        elif n==depth-1:
            depths+=[n]*2
        else:
            depths+=[n]
    data_lst = []
    beat_mini = 0
    # simulate(root_node,training_itterations)
    root_node = make_new_tree(model = model, add_noise = False)
    for n in range(num_games):
        result,data = play_mini(root_node,
                                as_x = as_x,
                                itterations = in_game_itterations,
                                print_true = False,
                                depth = depths[np.random.randint(0,len(depths))],
                                row_mult = row_mult,
                                col_mult = col_mult,
                                diag_mult = diag_mult,
                                len_4_vals=len_4_vals,
                                len_5_vals=len_5_vals,
                                len_6_vals=len_6_vals,
                                len_7_vals=len_7_vals)
        data_lst +=data
        beat_mini+=result
        for edge in root_node.true_root.edges_visited:
            edge.reset(reset_dir = False)
        root_node.true_root.edges_visited = set([])
    root_node = make_new_tree(model = model, add_noise = False)
    del root_node
    del model
    import tensorflow.keras.backend as K
    K.clear_session()
    return [beat_mini,data_lst]

def multi_play_mini_only_result(as_x,
                                len_n_vals_lst,
                                mult_lst,
                                depth = 3,
                                in_game_itterations = 400,
                                version_path = 'C:/Users/matth/python/connectfour/c4netST/V2/c4netST_versions/testversion'):
    len_4_vals = len_n_vals_lst[0]
    len_5_vals = len_n_vals_lst[1]
    len_6_vals = len_n_vals_lst[2]
    len_7_vals = len_n_vals_lst[3]
    model = make_eval_function(version_path)
    beat_mini = 0
    root_node = make_new_tree(model = model, add_noise = False)
    for mults in mult_lst:
        row_mult = mults[0]
        col_mult = mults[1]
        diag_mult = mults[2]
        result = play_mini(root_node,
                           as_x = as_x,
                           itterations = in_game_itterations,
                           print_true = False,
                           depth = depth,
                           row_mult = row_mult,
                           col_mult = col_mult,
                           diag_mult = diag_mult,
                           len_4_vals=len_4_vals,
                           len_5_vals=len_5_vals,
                           len_6_vals=len_6_vals,
                           len_7_vals=len_7_vals)[0]
        beat_mini+=result
        for edge in root_node.true_root.edges_visited:
            edge.reset(reset_dirc = False)
        root_node.true_root.edges_visited = set([])
    root_node = make_new_tree(model = model, add_noise = False)
    del root_node
    del model
    import tensorflow.keras.backend as K
    K.clear_session()
    return beat_mini


def watch_games_vs_mini(as_x,
                        len_n_vals_lst,
                        mult,
                        depth,
                        in_game_itterations,
                        version_path,
                        watch_full_games = True,
                        print_final_board = False):
    len_4_vals = len_n_vals_lst[0]
    len_5_vals = len_n_vals_lst[1]
    len_6_vals = len_n_vals_lst[2]
    len_7_vals = len_n_vals_lst[3]
    model = make_eval_function(version_path)
    root_node = make_new_tree(model = model, add_noise = False)
    row_mult = mult[0]
    col_mult = mult[1]
    diag_mult = mult[2]
    result = play_mini(root_node,
                       as_x = as_x,
                       itterations = in_game_itterations,
                       print_true = watch_full_games,
                       depth = depth,
                       row_mult = row_mult,
                       col_mult = col_mult,
                       diag_mult = diag_mult,
                       len_4_vals=len_4_vals,
                       len_5_vals=len_5_vals,
                       len_6_vals=len_6_vals,
                       len_7_vals=len_7_vals,
                       print_final_board = print_final_board)[0]
    del root_node
    del model
    import tensorflow.keras.backend as K
    K.clear_session()
    return result

def two_versions_play(current_version,
                      test_version,
                      current_version_as_x, 
                      itterations,
                      print_true=True):
    data = []
    turn_count =1
    if current_version_as_x:
        is_x=1
    else:
        is_x=0
    while not current_version.is_leaf or not test_version.is_leaf:
        if turn_count%2==is_x:
            chooser = 'Current c4netST'
            simulate(current_version,itterations)
            pi = current_version.pi
            move = pi.argmax()
            move = move+1
        else:
            chooser = 'test_version'
            simulate(test_version,itterations)
            pi = test_version.pi
            move = pi.argmax()
            move = move+1
        data.append([turn_count,current_version.split_board,pi])
        turn_count+=1
        if print_true:
            print('Current Board:\n')
            print(current_version.boardstr)
            player = chooser+' (Player '+current_version.player+')'
            statement = player +'s Turn'
            print('\n------------------------------------------------\n')
            print('')
            print(statement)
            print('')
            print(player,'chose ',move)
        for edge in current_version.edges:
            if edge.action == move:
                current_version = edge.make_target_node()
                break
        for edge in test_version.edges:
            if edge.action == move:
                test_version = edge.make_target_node()
                break
    if print_true:
        print(current_version.boardstr)
    for item in data:
        item.append(current_version.result)
    test_version_won = 0
    if current_version_as_x and current_version.result==2:
        test_version_won = 1
    elif not current_version_as_x and current_version.result==1:
        test_version_won=1
    del current_version
    del test_version
    return(test_version_won,data)

def multi_two_versions_play_only_result(current_version_path,
                                        current_version_as_x=True, 
                                        itterations=400,
                                        num_games = 50,
                                        add_noise = False):
    test_version_path = 'C:/Users/matth/python/connectfour/c4netST/V2/c4netST_versions/testversion'
    current_model = make_eval_function(current_version_path)
    test_model = make_eval_function(test_version_path)
    turn_count =1
    reset = True
    if current_version_as_x:
        is_x=1
    else:
        is_x=0
    test_version_won = 0
    for n in range(num_games):
        if n%25 ==0:
            current_version_root = make_new_tree(current_model,add_noise = add_noise)
            test_version_root = make_new_tree(test_model,add_noise = add_noise)
        current_version =  current_version_root
        test_version = test_version_root
        while not current_version.is_leaf or not test_version.is_leaf:
            print('current board')
            print(current_version.boardstr)
            if turn_count%2==is_x:
                print('cv turn')
                simulate(current_version,itterations)
                pi = current_version.pi
                move = pi.argmax()
                move = move+1
                print(move)
            else:
                print('tv turn')
                simulate(test_version,itterations)
                pi = test_version.pi
                move = pi.argmax()
                move = move+1
                print(move)
            turn_count+=1
            for edge in current_version.edges:
                if edge.action == move:
                    current_version = edge.make_target_node()
                    break
            for edge in test_version.edges:
                if edge.action == move:
                    test_version = edge.make_target_node()
                    break
        print('Final Board')
        print(current_version.boardstr)
        if current_version_as_x and current_version.result==2:
            test_version_won += 1
        elif not current_version_as_x and current_version.result==1:
            test_version_won += 1

        new_dirc = rng.dirichlet([1,2,3,4,5,6,7])
        half_edges = len(current_version_root.true_root.edges_visited)/2
        count = 0
        for edge in current_version_root.true_root.edges_visited:
            if count>half_edges:
                reset_dirc= reset
            else:
                reset_dirc = not reset
            edge.reset(reset_dirc =reset_dirc,new_dirc = new_dirc)
            count+=1
        current_version_root.true_root.edges_visited = set([])
        new_dirc = rng.dirichlet([1,2,3,4,5,6,7])
        half_edges = len(test_version_root.true_root.edges_visited)/2
        count = 0
        
        for edge in test_version_root.true_root.edges_visited:
            if count>half_edges:
                reset_dirc= reset
            else:
                reset_dirc = not reset
            edge.reset(reset_dirc =reset_dirc,new_dirc = new_dirc)
            count+=1
        test_version_root.true_root.edges_visited = set([])
        if n%5==4:
            reset = not reset
    current_version_root = make_new_tree(current_model,add_noise = add_noise)
    test_version_root = make_new_tree(test_model,add_noise = add_noise)
    current_version =  current_version_root
    test_version = test_version_root
    del current_version
    del test_version
    del current_version_root
    del test_version_root
    del current_model
    del test_model
    import tensorflow.keras.backend as K
    K.clear_session()
    return(test_version_won)




def self_train_many_games(model,
                          games=2,
                          trees =2,
                          in_game_itterations=100,
                          training_iterations=500,
                          print_true=True,
                          training_started = None):
    game_data = []
    toc2=tt()
    total_games = 0
    x_wins = 0
    o_wins = 0
    draws = 0
    avg_game_time = 0
    avg_train_time =0
    for n in range(trees):
        if print_true:
            print('\n---------------------------------------------------------\n')
        root_node = make_new_tree(model)
        if training_iterations>0:
            toc=tt()
            simulate(root_node,training_iterations)
            tic = tt()
            if print_true:
                avg_train_time = (n*avg_train_time+tic-toc)/(n+1)
                print('Training Time:',round(tic-toc,2))
                print('Average Training Time:',round(avg_train_time,2))
        toc = tt()
        for m in range(games):
            toc1 = tt()
            if m%10==9 and print_true:
                result,data = self_train(root_node=root_node,
                                  itterations=in_game_itterations,
                                  print_true=True)
            else:
                result,data = self_train(root_node=root_node,
                                  itterations=in_game_itterations,
                                  print_true=False)
            tic1=tt()
            total_games+=1
            if result == 1:
                x_wins+=1
            elif result ==2:
                o_wins +=1
            else:
                draws+=1
            if print_true:
                if training_started != None:
                    print('Training began at',training_started)
                print('Game:',(n,m))
                print('Game Time:',round(tic1-toc1,2))
                avg_game_time= ((total_games-1)*avg_game_time+tic1-toc1)/total_games
                print('Average Game Time:',round(avg_game_time,2))
                print('X win percent:',round(100*x_wins/total_games,2),'({}/{})'.format(x_wins,total_games))
                print('O win percent:',round(100*o_wins/total_games,2),'({}/{})'.format(o_wins,total_games))
                print('Draw percent:',round(100*draws/total_games,2),'({}/{})'.format(draws,total_games))
            game_data+=data
        tic =tt()
        if print_true:
            print('Itteration',n,'complete')
            print('Itteration Time:',round(tic-toc,2))
    tic2=tt()
    if print_true:
        print('Total Time:',tic2-toc2)
    del root_node
    return(game_data)


def multi_self_play(in_game_itterations = 200,
                    training_itterations = 2000,
                    games = 25,
                    version_path = 'C:/Users/matth/python/connectfour/c4netST/V2/c4netST_versions/c4netST_0-0'):
    model = make_eval_function(version_path)
    # simulate(root_node,training_itterations)
    data_lst = []
    reset = True
    for n in range(games):
        if n%25 == 0:
            root_node = make_new_tree(model = model)
        data_lst.append(self_train(root_node = root_node,
                                   itterations=in_game_itterations,
                                   print_true = False))
        new_dirc = rng.dirichlet([1,2,3,4,5,6,7])
        half_edges = len(root_node.true_root.edges_visited)/2
        count = 0
        for edge in root_node.true_root.edges_visited:
            if count>half_edges:
                reset_dirc= reset
            else:
                reset_dirc = not reset
            edge.reset(reset_dirc =reset_dirc,new_dirc = new_dirc)
            count+=1
        root_node.true_root.edges_visited = set([])
        if n%4:
            reset = not reset
    root_node = make_new_tree(model = model)
    del root_node
    del model
    import tensorflow.keras.backend as K
    K.clear_session()
    return(data_lst)



def make_single_boards(input_boards):
    boards = input_boards
    boards = boards[:,:,:,0]+boards[:,:,:,1]
    boards = np.moveaxis(boards,-1,1)
    
    return(boards)

def make_boards_scores(boards,scores):
    bs = np.zeros((boards.shape[0],7,7))
    for n in range(boards.shape[0]):
        bs[n,0,:] = scores[n]
        bs[n,1:,:] = boards[n]
        
    return(bs)

def make_boards_scores_with_pred(boards,scores,results,pred_scores,pred_results):
    bsp = np.zeros((boards.shape[0],10,7))
    for n in range(boards.shape[0]):
        bsp[n,0,:] = np.array([pred_results[n][0]]*7)
        bsp[n,1,:] = np.array([results[n]]*7)
        bsp[n,2,:] = pred_scores[n]
        bsp[n,3,:] = scores[n]
        bsp[n,4:,:] = boards[n]
        
    return(bsp)


if __name__=='__main__':
    pass
    # version_path = 'C:/Users/matth/python/connectfour/c4netST/V2/c4netST_versions/testversion'
    # model = make_eval_function(version_path)
    # for n in range(2):
    #     root_node = make_new_tree(model = model)
    #     toc =tt()
    #     itts = 20
    #     for n in range(itts):
    #         self_train(root_node,
    #                    )
    #     tic = tt()
    #     print((tic - toc)/itts)
    # data = multi_play_mini(True,
    #                     depth = 2,
    #                     in_game_itterations = 1,
    #                     training_itterations = 1,
    #                     num_games = 2,
    #                     version_path = 'C:/Users/matth/python/connectfour/c4netST/Version 2/c4netST_versions/c4netST_0_0')
    ##################################
    ###     Testing Run Times      ###
    ##################################
    # model = make_eval_function('C:/Users/matth/python/connectfour/c4netST_versions/c4netST_4-0')
    # version_path = 'C:/Users/matth/python/connectfour/c4netST/V2/c4netST_versions/c4netST_1-0'
    # model = make_eval_function(version_path)
    
    # itts = 200
    
    # time = 0
    # mm = 0
    # root_node = make_new_tree(model = model)
    # sims = 20
    # for n in range(itts):
    #     toc = tt()
    #     mm +=simulate(root_node,
    #              itterations=sims)
    #     tic = tt()
    #     time += tic - toc
    # print('simulations:',sims)
    # print('time:',time/itts)
    # print('Moves made:',mm/itts)
    
    
    
    # mm=0
    # time = 0
    # root_node = make_new_tree(model = model)
    # sims = 30
    # for n in range(itts):
    #     toc = tt()
    #     mm+=simulate(root_node,
    #              itterations=sims)
    #     tic = tt()
    #     time += tic - toc
    # print('simulations:',sims)
    # print('time:',time/itts)
    # print('Moves made:',mm/itts)
    
    
    # mm=0
    # time = 0
    # root_node = make_new_tree(model = model)
    # sims = 40
    # for n in range(itts):
    #     toc = tt()
    #     mm+=simulate(root_node,
    #              itterations=sims)
    #     tic = tt()
    #     time += tic - toc
    # print('simulations:',sims)
    # print('time:',time/itts)
    # print('Moves made:',mm/itts)
    
    
    
    # mm=0
    # time = 0
    # root_node = make_new_tree(model = model)
    # sims = 50
    # for n in range(itts):
    #     toc = tt()
    #     mm+=simulate(root_node,
    #              itterations=sims)
    #     tic = tt()
    #     time += tic - toc
    # print('simulations:',sims)
    # print('time:',time/itts)
    # print('Moves made:',mm/itts)
    
    # mm=0
    # time = 0
    # root_node = make_new_tree(model = model)
    # sims = 60
    # for n in range(itts):
    #     toc = tt()
    #     mm+=simulate(root_node,
    #              itterations=sims)
    #     tic = tt()
    #     time += tic - toc
    # print('simulations:',sims)
    # print('time:',time/itts)
    # print('Moves made:',mm/itts)
    
    
    
    # results,data = play_mini(root_node,
    #           as_x=True,
    #           itterations=1,
    #           print_true=True,
    #           depth=1,
    #           row_mult=1.5,
    #           col_mult=1,
    #           diag_mult=1.5)
    # game_times = {200:[],
    #               250:[],
    #               300:[],
    #               350:[],
    #               400:[],
    #               450:[],
    #               500:[]}
    
    # total_times = {200:0,
    #                250:0,
    #                300:0,
    #                350:0,
    #                400:0,
    #                450:0,
    #                500:0}

    # for key in game_times.keys():
    #     root_node = make_new_tree(model)
    #     print('')
    #     print('#'*50)
    #     print('')
    #     print(key,'itteration games:')
    #     toc = tt()
    #     x_won = 0 
    #     o_won = 0 
    #     draws = 0
    #     total_games = 0
    #     for n in range(25):
    #         toc1 = tt()
    #         result,data=self_train(model=None,root_node =root_node,itterations=key,print_true = False)
    #         tic1 = tt()
    #         total_games +=1
    #         if result == 1:
    #             x_won+=1
    #         elif result ==2:
    #             o_won+=1
    #         else:
    #             draws+=1
                
    #         print('Game',n+1,'Time:',round(tic1-toc1,2))
    #         print('Results:')
    #         print('X won',x_won,'of',total_games, '({}%)'.format(round(100*x_won/total_games,2)))
    #         print('O won',o_won,'of',total_games, '({}%)'.format(round(100*o_won/total_games,2)))
    #         print('Draws in',draws,'of',total_games, '({}%)'.format(round(100*draws/total_games,2)),'\n')
    #         game_times[key].append(round(tic1-toc1,2))
    #     tic = tt()
    #     print('-'*25,'\n')
    #     print(key,'itteration total time:',round(tic-toc,2))
    #     total_times[key] = round(tic-toc,2)
    

    # for key in total_times.keys():
    #     total_times_lsts[key].append(round(total_times[key],2))

    
