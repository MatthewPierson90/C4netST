"""
@author: Matthew

everything for the tree
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
from minimax_functions import has_won,new_board,minimax
rng = np.random.default_rng()
np.set_printoptions(suppress=True)
import pathlib
file_path = str(pathlib.Path(__file__).parent.resolve())+'\\c4netST_versions\\testversion'


def make_eval_function(to_load = file_path):
    """
    This function turns a keras model into a tensorflow lite model, and sets it to run on the CPU.  
    tflite models evaluate faster, and setting it to run on the cpu allows for easier parallel 
    data generation on a machine with a single CPU and GPU.

    Parameters
    ----------
    to_load : a string
        The path to the keras model to load

    Returns
    -------
    a tflite model.

    """
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
    del load_model
    del model
    del function
    del TensorSpec
    del model_eval
    del concrete_model_eval
    return(lite_model)


def open_data(file='base_layer.pkl'):
    """
    opens pickle files.
    
    """
    with open(file,'rb+') as pkl_file:
        layer1=pickle.load(pkl_file)
        pkl_file.close()
    return(layer1)

def save_data(data,file='base_layer.pkl'):
    """
    saves pickle files

    """
    with open(file,'wb+') as pkl_file:
        pickle.dump(data,pkl_file,-1)
        #print('saved')
        pkl_file.close()


def rlen(lst,start =0):
    """
    returns range(start,len(lst))

    Parameters
    ----------
    lst : 
        a list
    start : 
        the value to start at. The default is 0.

    Returns
    -------
    range(start,len(lst))
    """
    return range(start,len(lst))


def board_str(board):
    """
    turns a board into a string.
    """
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



def available_moves(board):
    """
    determines the available moves
    """
    moves = np.zeros(7)
    for n in range(7):
        if board[n][0]==0:
            moves[n]=1
    return(moves.astype(np.int32))



def make_move(board, col, player):
    """
    makes a move on the board

    Parameters
    ----------
    board : np.array
        the board
    col : int
        the column the player will drop their piece down
    player : int
        1 or 2.

    Returns
    -------
    The new board.

    """
    for n in range(6):
        if n == 5:
            board[col-1][n] = player
            break
        elif board[col-1][n+1]!= 0:
            board[col-1][n] = player
            break
    return(board)


class node_class(object):
    """
    A game board, with all of the possible moves that can be made, 
    scores for all of the possible moves that can be made,
    the tflite model used for evaluating the boards.
    """
    def __init__(self,
                 state,
                 model_eval_func,
                 dirc = None,
                 player='X',
                 player_num = 1,
                 add_noise = True,
                 true_root = None):
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
        self.moves = available_moves(state)
        self.player = player
        self.player_num = player_num
        self.model_eval_func = model_eval_func
        self.is_leaf = False
        if has_won(state,1):
            self.is_leaf = True
            self.result = 1
        elif has_won(state,2):
            self.is_leaf = True
            self.result = 2
        elif max(self.moves)==0:
            self.is_leaf = True
            self.result = 0
        if not self.is_leaf:
            model_P,model_V = self.model_eval_func.predict(self.split_board)
            self.model_V = model_V[0]
            self.model_P = model_P
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

    
    def __str__(self):
        return 'Board:\n{0} \nPlayer: {1}\n'.format(self.boardstr,self.player)
        
    def __repr__(self):
        return str(self)


class edge_class(object):
    """
    the moves that can be made from a node,
    the new nodes location,
    all kinds of good stuffs.
    """
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
        self.W = 0
        self.Pi = 0
        preU=self.C*self.P
        self.U = preU #base ratio
        self.av = self.Q+self.U #action value
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
    def __str__(self):
        return 'board:\n{0}making move:{1} \nPlayer:{2}\n'.format(self.boardstr,self.action,self.player)
        
    def __repr__(self):
        return str(self)



def tree_node_edge_count(root_node):
    """
    counts the number of nodes visited and edges seen.

    Parameters
    ----------
    root_node : original node of the tree

    Returns
    -------
    the number of unique nodes and edges seen.

    """
    node_count = 1
    edge_count = 0
    if not root_node.is_leaf:
        for edge in root_node.edges:
            edge_count += 1
            if edge.made_target_node == 1:
                to_add_node_count,to_add_edge_count=tree_node_edge_count(edge.target_node)
                node_count+=to_add_node_count
                edge_count+=to_add_edge_count
        return(node_count,edge_count)
    else:
        return(1,0)


def print_node_edge_info(node,print_all = False,av=True,W=False,N=False,layer_N=False, P=False,U=False,Q=False):
    """
    prints all of the info for a given node and its source edges. 

    Parameters
    ----------
    node : Node
        a node!
    print_all : boolean, optional
        print all info? The default is False.
    av : boolean, optional
        print action value info for each edge? The default is True.
    W : boolean, optional
        print w (wins - losses going down this edge) info? The default is False.
    N : boolean, optional
        print N (edge visit count) info? The default is False.
    layer_N : boolean, optional
        print (node visit count) info? The default is False.
    P : boolean, optional
        print P info? The default is False.
    U : boolean, optional
        print U by visit count info? The default is False.
    Q : boolean, optional
        print Q info? The default is False.

    Returns
    -------
    None.  Prints desired info

    """
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
    """
    makes a new tree, aka makes the root node.  

    Parameters
    ----------
    model : keras model
    add_noise : Boolean
        adds dirac delta noise to promote exploration. The default is True.

    Returns
    -------
    a root node.

    """
    return(node_class(new_board(),model,add_noise=add_noise))


def simulate(root_node,
             itterations = 1000,
             reset_edges = True):
    """
    makes a number of unique simulated moves to adjust the edge and node values

    Parameters
    ----------
    root_node : a root node
    itterations : int, optional
        The number of itterations to make
    reset_edges : boolean, optional
        Keeps track of the edges visited, after the game, 
        the edges will be kept but their info cleared. 
        The default is True.

    Returns
    -------
    The total number of moves made.

    """
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



def self_train(model = None,
               root_node = None,
               itterations = 1000,
               print_true = True,
               reset_edges = False,
               print_final = True):
    """
    The MCTS model plays itself. Note, one of model or rootnode must not be none

    Parameters
    ----------
    model : tflite model, optional
        the model used to evaluate the board state. The default is None.
    root_node : root node, optional
        a root node. The default is None.
    itterations : int, optional
        num itterations to make. The default is 1000.
    print_true : boolean, optional
        each game board. The default is True.
    reset_edges : boolean, optional
        if true, resets all of the visited 
        edge and node info. The default is False.
    print_final : TYPE, optional
        If true and print_true is false, then the final board is printed.
        The default is True.

    Returns
    -------
    the result and game data.

    """
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
    """
    The MCTS model plays itself, but doesn't collect game data.
    Note, one of model or rootnode must not be none

    Parameters
    ----------
    model : tflite model, optional
        the model used to evaluate the board state. The default is None.
    root_node : root node, optional
        a root node. The default is None.
    itterations : int, optional
        num itterations to make. The default is 1000.
    print_true : boolean, optional
        each game board. The default is True.
    reset_edges : boolean, optional
        if true, resets all of the visited 
        edge and node info. The default is False.
    print_final : TYPE, optional
        If true and print_true is false, then the final board is printed.
        The default is True.

    Returns
    -------
    The total simulation time and number of moves made

    """
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


def play_mini(root_node,
              as_x = True,
              itterations = 100,
              print_true = True,
              depth = 4,
              row_mult = 1.5,
              col_mult = 1.,
              diag_mult = 1.5,
              len_4_vals = open_data('array_n_values/len_4_values.pkl'),
              len_5_vals = open_data('array_n_values/len_5_values.pkl'),
              len_6_vals = open_data('array_n_values/len_6_values.pkl'),
              len_7_vals = open_data('array_n_values/len_7_values.pkl'),
              print_final_board = False):
    """
    has MCTS model play a minimax.
    Parameters
    ----------
    root_node : root node, optional
        a root node. The default is None.
    itterations : int, optional
        num itterations to make. The default is 1000.
    print_true : boolean, optional
        each game board. The default is True.
    depth : int, optional
        minimax depth, default is 4.
    row_mult : float, optional
        row multiplier, if high minimax will prioritize rows. The default is 1.5.
    col_mult : float, optional
        column multiplier, if high minimax will prioritize columns. The default is 1.
    diag_mult : float, optional
        diagonal multiplier, if high minimax will prioritize diagonals. The default is 1.5.
    len_4_vals : np array, optional
        Precomputed values, if you can figure it out, you can change how much minimax 
        prioritizes three streaks  over two streaks. 
        The default is open_data('array_n_values/len_4_values.pkl').
    len_5_vals : np array, optional
        Precomputed values. The default is open_data('array_n_values/len_5_values.pkl').
    len_6_vals : np array, optional
        Precomputed values. The default is open_data('array_n_values/len_6_values.pkl').
    len_7_vals : np array, optional
        Precomputed values. The default is open_data('array_n_values/len_7_values.pkl').
    print_final : TYPE, optional
        If true and print_true is false, then the final board is printed.
        The default is True.

    Returns
    -------
    The winner and game data

    """
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
    if as_x and root_node.result == 1:
        beat_mini = 1
    elif not as_x and root_node.result == 2:
        beat_mini = 1
    # del root_node
    return(beat_mini,data)

def multi_play_mini(as_x,
                    len_4_vals,
                    len_5_vals,
                    len_6_vals,
                    len_7_vals,
                    depth,
                    in_game_itterations = 1000,
                    training_itterations = 1,
                    num_games = 25,
                    version_path = file_path):
    """
    Intended to be used with multiprocessing. The multiprocessing call is in the training pipeline file.

    Parameters
    ----------
    as_x : boolean
        if True, MCTS plays as X
    len_4_vals : np array
        Precomputed values, will be loaded prior to the multiprocessing call.
    len_5_vals : np array
        Precomputed values, will be loaded prior to the multiprocessing call.
    len_6_vals : np array
        Precomputed values, will be loaded prior to the multiprocessing call.
    len_7_vals : np array
        Precomputed values, will be loaded prior to the multiprocessing call.
    depth : int
        minimax depth
    in_game_itterations : int, optional
        simulation itterations made in game. The default is 200.
    training_itterations : int, optional
        simulation itterations made prior to a game.  
        Idea is that the Tree can self train before playing the minimax. The default is 1.
    num_games : int, optional
        The number of games each tree will play. The default is 25.
    version_path : string, optional
        the path to the model. The default is file_path.

    Returns
    -------
    list
        game data

    """
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



def multi_play_mini_only_result(as_x,
                                len_n_vals_lst,
                                mult_lst,
                                depth = 3,
                                in_game_itterations = 400,
                                version_path = file_path):
    """
    Used for testing the model against a set of minimax algorithms. 
    Intended to be used with multiprocessing, 
    the multiprocessing call is in the training pipeline file.

    Parameters
    ----------
    as_x : boolean
        if True, MCTS plays as X
    len_n_vals_lst : list
        List of precomputed values, will be loaded prior to the multiprocessing call.
    mult_lst : list
        list of multipliers for minimax
    depth : int, optional
        . The default is 3.
    in_game_itterations : int, optional
        simulation itterations made in game. The default is 400.
    version_path : string, optional
        the path to the model. The default is file_path.

    Returns
    -------
    beat_mini : int
        whether or not it beat the minimax.

    """
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
    """
    Watch a game vs the minimax, to help spot errors.
    """
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
    """
    Two versions play each other. For testing and data generation
    """
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
                                        add_noise = False,
                                        test_version_path = file_path):
    """
    two versions play each other, intended to be used with multiprocessing. For testing
    """
    current_model = make_eval_function(current_version_path)
    test_model = make_eval_function(test_version_path)
    turn_count = 1
    reset = True
    if current_version_as_x:
        is_x = 1
    else:
        is_x = 0
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
            if turn_count%2 == is_x:
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
    """
    multiple trees play multiple games.  For data generation on a single processor.
    """
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
                    version_path = file_path):
    """
    self play with multiprocessing.  For data generation.
    """
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



# def make_single_boards(input_boards):
#     boards = input_boards
#     boards = boards[:,:,:,0]+boards[:,:,:,1]
#     boards = np.moveaxis(boards,-1,1)
    
#     return(boards)

# def make_boards_scores(boards,scores):
#     bs = np.zeros((boards.shape[0],7,7))
#     for n in range(boards.shape[0]):
#         bs[n,0,:] = scores[n]
#         bs[n,1:,:] = boards[n]
        
#     return(bs)

# def make_boards_scores_with_pred(boards,scores,results,pred_scores,pred_results):
#     bsp = np.zeros((boards.shape[0],10,7))
#     for n in range(boards.shape[0]):
#         bsp[n,0,:] = np.array([pred_results[n][0]]*7)
#         bsp[n,1,:] = np.array([results[n]]*7)
#         bsp[n,2,:] = pred_scores[n]
#         bsp[n,3,:] = scores[n]
#         bsp[n,4:,:] = boards[n]
        
#     return(bsp)
