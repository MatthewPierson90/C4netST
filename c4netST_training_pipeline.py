"""
@author: Matthew

training pipeline functions
"""

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(50)
# import winsound
import numpy as np
import pathlib
file_path = str(pathlib.Path(__file__).parent.resolve())+'\\'
from mcts_with_c4net import open_data,save_data,play_mini ,simulate,make_new_tree
from mcts_with_c4net import make_eval_function,self_train_many_games,two_versions_play
from mcts_with_c4net import multi_self_play, multi_play_mini, watch_games_vs_mini
from mcts_with_c4net import multi_play_mini_only_result, multi_two_versions_play_only_result
from CNNdata import mcts_data,shuffle_mcts_data
# from tensorflow.keras.models import load_model
import random
import time
from datetime import datetime
from multiprocessing import Pool
print(datetime.now())
tt=time.time


def clear_numba(file_path = file_path):
    """Cached numba functions freak out over the slieghtest change.  
    This function deletes them."""
    lst = os.listdir(file_path+'__pycache__')
    for file in lst:
        if 'Using_numba' in file:
            path = file_path+'__pycache__/'+file
            os.remove(path)


def get_versions(file_path = file_path):
    """Gives you info on the current version and all of the previous versions"""
    from mcts_with_c4net import open_data
    current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
    versions = open_data(file_path+'c4netST_versions/versions.pkl')
    return(current_version,versions)

# if __name__ == '__main__':
#     current_version,versions = get_versions()


def c4netST_training_pipeline(training_loops=1,
                              in_game_iterations_vs_self = 1000,
                              training_iterations=1,
                              num_self_games_per_tree=250,
                              num_trees=16,
                              batch_size = 1024,
                              epochs = 20,
                              print_true =True,
                              use_multi_self = True, #if True do behind if __name__ == __main__ 
                              use_multi_test = True,
                              num_proc_multi_self = 8,
                              current_version = None,#change this for testing only!
                              num_test_games = 100,#change this for testing only!
                              file_path = file_path): 
    """Generates data, creates and fits a new model after a specified number of games,
    tests the new model vs the old model and vs minimax.
    Uses the new model to generate data if it preformed well enough. Does the loop again"""
    time = datetime.now()
    training_started = time.strftime("%H:%M:%S")
    self_time = 0
    training_time = 0
    testing_time = 0
    update_time = 0
    print('Training started at', training_started)
    if training_loops == 1:
        print('Training for',training_loops,'loop!')
    else:
        print('Training for',training_loops,'loops!')
    toc = tt()
    for n in range(training_loops):
        print('\n----------------------------------------------------------\n')
        print('Beginning loop',n+1,'of',training_loops,'\n')
        if use_multi_self:
            self_toc = tt()
            multi_make_self_data(num_self_games_per_tree=num_self_games_per_tree,
                                 num_trees=num_trees,
                                 in_game_iterations = in_game_iterations_vs_self,
                                 training_iterations = training_iterations,
                                 training_started = training_started,
                                 current_version = current_version,
                                 num_proc = num_proc_multi_self) #8 just barely maxes out memory doing 400 simulations per move 25 games per tree, may have issues with 40 games per tre
        else:
            
            self_toc = tt()
            make_self_data(num_self_games_per_tree=num_self_games_per_tree,
                           num_trees=num_trees,
                           in_game_iterations=in_game_iterations_vs_self,
                           training_iterations=training_iterations,
                           print_true = print_true,
                           training_started = training_started,
                           current_version = current_version)
        self_tic = tt()
        self_time += self_tic - self_toc
        training_toc = tt()
        times_through = 0
        vs_mini_percent = 0
        vs_current_percent = 0
        while vs_mini_percent <85 and vs_current_percent <55 and times_through<2:
            print('Times Through:',times_through)
            gameboards_trained_on = make_and_fit_transfer_model(epochs = epochs, 
                                                                batch_size = batch_size)
            training_tic = tt()
            training_time += training_tic-training_toc
            if use_multi_test:
                testing_toc = tt()
                vs_mini_percent, results_lst_mini=multi_testnet_vs_mini(in_game_iterations = in_game_iterations_vs_self)
                print('Percent won vs mini:',vs_mini_percent)
                print('Mini Results lst (32 games each):',results_lst_mini)
                print('Percent won as X:',round(100*sum(results_lst_mini[:4])/128,2))
                print('Percent won as O:',round(100*sum(results_lst_mini[4:])/128,2))
                vs_current_percent, results_lst_current = multi_testnet_vs_current(in_game_iterations = in_game_iterations_vs_self)
                print('Percent won vs current:',vs_current_percent)
                print('Results vs current version (25 games each):',results_lst_current)
                print('Percent won as X:',sum(results_lst_current[:4]))
                print('Percent won as O:',sum(results_lst_current[4:]))
            else:
                testing_toc = tt()
                vs_mini_percent = get_win_percent_testnet_vs_mini(training_iterations = training_iterations,
                                                                      in_game_iterations = in_game_iterations_vs_self,
                                                                      num_games = num_test_games,
                                                                      print_true = print_true)
                
                vs_current_percent = get_win_percent_vs_current_version(iterations=in_game_iterations_vs_self)
            testing_tic = tt()
            testing_time += testing_tic-testing_toc
            times_through+=1
        update_toc = tt()
        update_version_and_data(vs_current_percent = vs_current_percent,
                                vs_mini_percent = vs_mini_percent,
                                gameboards_trained_on = gameboards_trained_on,
                                results_lst_mini = results_lst_mini,
                                results_lst_current = results_lst_current)
        update_tic = tt()
        update_time += update_tic-update_toc
        import tensorflow.keras.backend as K
        K.clear_session()
            
        tic = tt()
        if training_loops == 1:
            print('Completed',training_loops,'training loop!')
        else:
            print('Completed',training_loops,'training loops!')
        print('Total Time:',round(tic-toc,2))
        print('Self Training Time:',self_time)
        print('Fitting testnet Time:',training_time)
        print('Testing testnet Time:',testing_time)
        print('Update Time:',update_time)



def make_self_data(num_self_games_per_tree=10,
                   num_trees=50,
                   in_game_iterations=200,
                   training_iterations=2000,
                   print_true = True,
                   training_started = None,
                   current_version = None,
                   file_path = file_path):
    """Generates data by having the mcts play itself. For a single processor"""
    if current_version == None:
        current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
    version_path = file_path+'c4netST_versions/'+current_version['name']
    model = make_eval_function(version_path)
    #Takes about 70 minutes
    print('Beginning Self Training Loop')
    data_self=self_train_many_games(model,
                                    games= num_self_games_per_tree,
                                    trees =num_trees,
                                    in_game_iterations=in_game_iterations,
                                    training_iterations=training_iterations,
                                    training_started = training_started)
    data_version = '{}_{}_{}'.format(current_version['minimax_level_beat'],
                                     current_version['mini_level_version'],
                                     current_version['training_cycles'])
    data_name_self = file_path+'c4netST_data/{}_{}_{}.pkl'.format('self','play',data_version)
    save_data(data_self,data_name_self)
    del model

def multi_make_self_data(num_self_games_per_tree=25,
                         num_trees=40,
                         in_game_iterations=200,
                         training_iterations=2000,
                         training_started = None,
                         current_version = None,
                         num_proc = 8,
                         file_path = file_path):
    """Generates data by having the mcts play itself. For multiprocessing"""
    if current_version == None:
        current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
    version_path = file_path+'c4netST_versions/'+current_version['name']
    inputs = [(in_game_iterations,
               training_iterations,
               num_self_games_per_tree,
               version_path) for n in range(num_trees)]
    #Takes about 70 minutes
    time = datetime.now()
    self_loop_started = time.strftime("%H:%M:%S")
    print('Beginning Self Training Loop')
    print('Training began at',training_started)
    print('Self Training Loop began at ',self_loop_started)
    toc = tt()
    with Pool(num_proc) as p:
        data_lsts = p.starmap(multi_self_play,inputs)
    tic = tt()
    print('Getting data for',num_trees*num_self_games_per_tree,'games took', tic-toc)
    data_self_with_result =[]
    data_self = []
    for data in data_lsts:
        data_self_with_result+=data
    for data in data_self_with_result:
        data_self+=data[1]
    data_version = '{}_{}_{}'.format(current_version['minimax_level_beat'],
                                     current_version['mini_level_version'],
                                     current_version['training_cycles'])
    data_name_self = file_path+'c4netST_data/{}_{}_{}.pkl'.format('self','play',data_version)
    save_data(data_self,data_name_self)



def multi_make_mini_data(mini_games_per_root = 50,
                         num_roots = 20,
                         in_game_iterations = 200,
                         training_iterations = 2000,
                         training_started = None,
                         current_version = None,
                         num_proc = 8,
                         file_path = file_path):
    """Generates data by having the mcts play a set of minimax algorithms. For multiprocessing.
    Not used in this pipeline"""
    file_path = 'C:/Users/matth/python/connectfour/c4netST/V2/'
    clear_numba()
    if current_version == None:
        current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
    version_path = file_path+'c4netST_versions/'+current_version['name']
    inputs = []
    as_x = True
    len_4_vals = open_data('array_n_values/len_4_values.pkl')
    len_5_vals = open_data('array_n_values/len_5_values.pkl')
    len_6_vals = open_data('array_n_values/len_6_values.pkl')
    len_7_vals = open_data('array_n_values/len_7_values.pkl')
    for n in range(num_roots):
        to_append = [as_x,
                     len_4_vals,
                     len_5_vals,
                     len_6_vals,
                     len_7_vals,
                     current_version['minimax_level_beat']+1,
                     in_game_iterations,
                     training_iterations,
                     mini_games_per_root,
                     version_path]
        inputs.append(to_append)
        as_x = not as_x
    #Takes about 70 minutes
    time = datetime.now()
    self_loop_started = time.strftime("%H:%M:%S")
    print('Beginning Mini Training Loop')
    print('Training began at', training_started)
    print('Mini Training Loop began at ',self_loop_started)
    print('Training vs mini level:',current_version['minimax_level_beat']+1)
    toc = tt()
    with Pool(num_proc) as p:
        data_lsts = p.starmap(multi_play_mini,inputs)
    tic = tt()
    print('Getting data for', num_roots * mini_games_per_root, 'games took', tic-toc)
    data_mini = []
    beat_mini = 0
    for data in data_lsts:
        beat_mini+=data[0]
        data_mini+=data[1]
    data_version = '{}_{}_{}'.format(current_version['minimax_level_beat'],
                                      current_version['mini_level_version'],
                                      current_version['training_cycles'])
    data_name_mini = file_path+'c4netST_data/{}_{}_{}.pkl'.format('mini','play',data_version)
    save_data(data_mini,data_name_mini)
    print(beat_mini)
    return beat_mini


def make_mini_data(mini_games=1000,
                   in_game_iterations=200,
                   training_iterations=2000,
                   print_true =True,
                   training_started = None,
                   current_version = None,
                   file_path = file_path):
    """Generates data by having the mcts play a set of minimax algorithms. For a single processor.
    Not used in this pipeline"""
    clear_numba()
    print('Beginning Mini Training Loop \n')
    as_x=False
    average_game_time = 0
    toc3=tt()
    won_vs_mini=0
    won_as_x = 0
    total_x_games = 0
    won_as_o = 0
    total_o_games = 0
    if current_version == None:
        current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
    version_path = file_path+'c4netST_versions/'+current_version['name']
    model = make_eval_function(version_path)
    depth = current_version['minimax_level_beat']+1
    root_node = make_new_tree(model, add_noise= False)
    print('Current Version:',current_version['name'])
    print('Minimax Level:',current_version['minimax_level_beat']+1,'\n')
    toc2=tt()
    simulate(root_node,training_iterations)
    tic2=tt()
    print('Training Time:',tic2-toc2)
    average_train_time = 0
    times_trained = 0
    games_data =[]
    len_4_vals = open_data(file_path+'array_n_values/len_4_values.pkl')
    len_5_vals = open_data(file_path+'array_n_values/len_5_values.pkl')
    len_6_vals = open_data(file_path+'array_n_values/len_6_values.pkl')
    len_7_vals = open_data(file_path+'array_n_values/len_7_values.pkl')
    for n in range(mini_games):
        row_mult=(random.random()+random.randint(1,2))
        col_mult=(random.random()+random.randint(1,2))
        diag_mult=(random.random()+random.randint(1,2))
        row_mult=row_mult
        col_mult=col_mult
        diag_mult = diag_mult
        toc1=tt()
        if n%10==9 and print_true:
            toc1=tt()
            won,data=play_mini(root_node,
                               as_x=as_x,
                               iterations=in_game_iterations,
                               print_true=True,
                               depth=depth+1,
                               row_mult=row_mult,
                               col_mult=col_mult,
                               diag_mult = diag_mult,
                               len_4_vals=len_4_vals,
                               len_5_vals=len_5_vals,
                               len_6_vals=len_6_vals,
                               len_7_vals=len_7_vals)
            won_vs_mini+=won
            if as_x:
                won_as_x+=won
                total_x_games+=1
            else:
                won_as_o+=won
                total_o_games+=1
            tic1=tt()
            average_game_time = (average_game_time*n+tic1-toc1)/(n+1)
            print('')
            print('-'*50,'\n')
            if training_started!= None:
                print('Training began at',training_started)
            print('Game', n,'Time:',round(tic1-toc1,2))
            print('Average Game Time:', round(average_game_time,2))
            print('Games Remaining:',mini_games-n-1)
            print('Total Time:',round(tt()-toc3,2))
            percent = round(100*won_vs_mini/(n+1),2)
            print('Won:',won_vs_mini,'of',n+1,'({}%)'.format(percent))
            if total_x_games >0:
                percent = round(100*won_as_x/total_x_games,2)
                print('Won as X:',won_as_x,'of',total_x_games,'({}%)'.format(percent))
            if total_o_games > 0:
                percent = round(100*won_as_o/total_o_games,2)
                print('Won as O:',won_as_o,'of',total_o_games,'({}%)'.format(percent))
        else:
            toc1=tt()
            won,data=play_mini(root_node,
                               as_x=as_x,
                               iterations=in_game_iterations,
                               print_true=False,
                               depth=depth+1,
                               row_mult=row_mult,
                               col_mult=col_mult,
                               diag_mult = diag_mult,
                               len_4_vals=len_4_vals,
                               len_5_vals=len_5_vals,
                               len_6_vals=len_6_vals,
                               len_7_vals=len_7_vals)
            tic1=tt()
            won_vs_mini+=won
            if as_x:
                won_as_x+=won
                total_x_games+=1
            else:
                won_as_o+=won
                total_o_games+=1
            average_game_time = (average_game_time*n+tic1-toc1)/(n+1)
            if print_true:
                print('')
                print('-'*50,'\n')
                print('Game', n,'Time:',round(tic1-toc1,2))
                print('Average Game Time:', round(average_game_time,2))
                print('Games Remaining:',mini_games-n-1)
                print('Total Time:',round(tt()-toc3,2))
                percent = round(100*won_vs_mini/(n+1),2)
                print('Won:',won_vs_mini,'of',n+1,'({}%)'.format(percent))
                if total_x_games >0:
                    percent = round(100*won_as_x/total_x_games,2)
                    print('Won as X:',won_as_x,'of',total_x_games,'({}%)'.format(percent))
                if total_o_games > 0:
                    percent = round(100*won_as_o/total_o_games,2)
                    print('Won as O:',won_as_o,'of',total_o_games,'({}%)'.format(percent))
        games_data+=data
        if n%50 == 49:
            root_node = make_new_tree(model, add_noise = False)
            toc2=tt()
            simulate(root_node,training_iterations)
            tic2=tt()
            times_trained += 1
            average_train_time = ((times_trained-1)*average_train_time+tic2-toc2)/times_trained
            print('Training Time:',tic2-toc2)
            as_x=not as_x
    percent = round(100*won_vs_mini/mini_games,2)
    print('Won:',won_vs_mini,'of',mini_games,'({}%)'.format(percent))
    if total_x_games >0:
        percent = round(100*won_as_x/total_x_games,2)
        print('Won as X:',won_as_x,'of',total_x_games,'({}%)'.format(percent))
    if total_o_games >0:
        percent = round(100*won_as_o/total_o_games,2)
        print('Won as O:',won_as_o,'of',total_o_games,'({}%)'.format(percent))
    data_version = '{}_{}_{}'.format(current_version['minimax_level_beat'],
                                     current_version['mini_level_version'],
                                     current_version['training_cycles'])
    data_name = file_path+'c4netST_data/{}_{}_{}.pkl'.format('mini','play',data_version)
    save_data(games_data,data_name)
    del root_node
    del model
    return(won_vs_mini,mini_games)



def get_win_percent_testnet_vs_mini(training_iterations = 1,
                                    in_game_iterations=400,
                                    num_games = 100,
                                    print_true = True,
                                    depth = None,
                                    test_version_path = None,
                                    file_path = file_path):
    """For testing the test version vs minimax. Single processor"""
    clear_numba()
    average_game_time = 0
    as_x = True
    won_vs_mini=0
    won_as_x = 0
    total_x_games = 0
    won_as_o = 0
    total_o_games = 0
    current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
    if test_version_path == None:
        test_version_path = file_path+'c4netST_versions/testversion'
    print('\n------------------------------------------------------------\n')
    print('Current Version:',current_version['name'])
    len_4_vals = open_data(file_path+'array_n_values/len_4_values.pkl')
    len_5_vals = open_data(file_path+'array_n_values/len_5_values.pkl')
    len_6_vals = open_data(file_path+'array_n_values/len_6_values.pkl')
    len_7_vals = open_data(file_path+'array_n_values/len_7_values.pkl')
    model = make_eval_function(test_version_path)
    if depth == None:
        depth = current_version['minimax_level_beat']+1
    print('Testing Minimax Level:',depth)
    root_node = make_new_tree(model, add_noise = False)
    simulate(root_node,training_iterations)
    for n in range(num_games):
        if n%50==0 and n!=0:
                root_node = make_new_tree(model, add_noise = False)
                simulate(root_node,training_iterations)
                as_x=not as_x
        row_mult=(random.random()+random.randint(1,2))
        col_mult=(random.random()+random.randint(1,2))
        diag_mult=(random.random()+random.randint(1,2))
        if n%10==9 and print_true:
            toc1 = tt()
            won,data=play_mini(root_node,
                                    as_x=as_x,
                                    iterations=in_game_iterations,
                                    print_true=True,
                                    depth=depth+1,
                                    row_mult=row_mult,
                                    col_mult=col_mult,
                                    diag_mult = diag_mult,
                                    len_4_vals=len_4_vals,
                                    len_5_vals=len_5_vals,
                                    len_6_vals=len_6_vals,
                                    len_7_vals=len_7_vals)
            won_vs_mini+=won
            if as_x:
                won_as_x+=won
                total_x_games+=1
            else:
                won_as_o+=won
                total_o_games+=1
            tic1=tt()
            average_game_time = (average_game_time*n+tic1-toc1)/(n+1)
            print('Game', n,'Time:',round(tic1-toc1,2))
            print('Average Game Time:', round(average_game_time,2))
            print('Games Remaining:',num_games-n-1)
            percent = round(100*won_vs_mini/(n+1),2)
            print('Won:',won_vs_mini,'of',n+1,'({}%)'.format(percent))
            if total_x_games >0:
                percent = round(100*won_as_x/total_x_games,2)
                print('Won as X:',won_as_x,'of',total_x_games,'({}%)'.format(percent))
            if total_o_games > 0:
                percent = round(100*won_as_o/total_o_games,2)
                print('Won as O:',won_as_o,'of',total_o_games,'({}%)'.format(percent))
        else:
            toc1=tt()
            won,data=play_mini(root_node,
                                    as_x=as_x,
                                    iterations=in_game_iterations,
                                    print_true=False,
                                    depth=depth+1,
                                    row_mult=row_mult,
                                    col_mult=col_mult,
                                    diag_mult = diag_mult,
                                    len_4_vals=len_4_vals,
                                    len_5_vals=len_5_vals,
                                    len_6_vals=len_6_vals,
                                    len_7_vals=len_7_vals)
            won_vs_mini+=won
            if as_x:
                won_as_x+=won
                total_x_games+=1
            else:
                won_as_o+=won
                total_o_games+=1
            tic1=tt()
            average_game_time = (average_game_time*n+tic1-toc1)/(n+1)
            if print_true:
                print('')
                print('-'*50,'\n')
                
                print('Game', n,'Time:',round(tic1-toc1,2))
                print('Average Game Time:', round(average_game_time,2))
                print('Games Remaining:',num_games-n-1)
                percent = round(100*won_vs_mini/(n+1),2)
                print('Won:',won_vs_mini,'of',n+1,'({}%)'.format(percent))
                if total_x_games >0:
                    percent = round(100*won_as_x/total_x_games,2)
                    print('Won as X:',won_as_x,'of',total_x_games,'({}%)'.format(percent))
                if total_o_games > 0:
                    percent = round(100*won_as_o/total_o_games,2)
                    print('Won as O:',won_as_o,'of',total_o_games,'({}%)'.format(percent))
    percent = round(100*won_vs_mini/(n+1),2)
    print('Won',won_vs_mini,'of',n+1,'({}%)'.format(percent))
    if total_x_games >0:
        percent = round(100*won_as_x/total_x_games,2)
        print('Won as X:',won_as_x,'of',total_x_games,'({}%)'.format(percent))
    if total_o_games > 0:
        percent = round(100*won_as_o/total_o_games,2)
        print('Won as O:',won_as_o,'of',total_o_games,'({}%)'.format(percent))
    del model
    del root_node
    testnet_win_percent = round(100*won_vs_mini/num_games,2)
    return(testnet_win_percent)




def get_win_percent_vs_current_version(version_path=None,
                                       iterations=400,
                                       num_games = 100,
                                       file_path = file_path):
    """For testing the test version vs current version. Single processor"""
    as_x = True
    test_version_won=0
    if version_path == None:
        current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
        version_path = file_path+'c4netST_versions/'+current_version['name']
    current_model = make_eval_function(version_path)
    test_version_path = file_path+'c4netST_versions/testversion'
    test_model = make_eval_function(test_version_path)
    current_model_root_node = make_new_tree(current_model,add_noise = False)
    test_model_root_node = make_new_tree(test_model,add_noise = False)
    print('\n------------------------------------------------------------\n')
    for n in range(num_games):
        if n%10==9:
            won,data=two_versions_play(current_model_root_node,
                                       test_model_root_node,
                                       current_version_as_x = as_x, 
                                       iterations = iterations,
                                       print_true = True)
            test_version_won+=won
            percent = round(100*test_version_won/(n+1),2)
            print('Test version won',test_version_won,'of',n+1,'({}%)'.format(percent))
        else:
            won,data=two_versions_play(current_model_root_node,
                                       test_model_root_node,
                                       current_version_as_x = as_x, 
                                       iterations = iterations,
                                       print_true = False)
            test_version_won+=won
            percent = round(100*test_version_won/(n+1),2)
            print('\nTest version won',test_version_won,'of',n+1,'({}%)'.format(percent),'\n')
        if n%50 == 49:
            current_model_root_node = make_new_tree(current_model,add_noise = False)
            test_model_root_node = make_new_tree(test_model,add_noise = False)
            as_x=not as_x
    percent = round(100*test_version_won/num_games,2)
    print('Test version won',test_version_won,'of',n+1,'({}%)'.format(percent))
    del current_model
    del test_model
    del current_model_root_node
    del test_model_root_node
    import tensorflow.keras.backend as K
    K.clear_session()
    return(percent)



def multi_testnet_vs_mini(in_game_iterations = 400,
                          num_proc = 8,
                          current_version = None,
                          depth = None,
                          test_version = 'testversion',
                          file_path = file_path):
    """For testing the test version vs minimax. For multiprocessing"""
    clear_numba()
    len_4_vals = open_data(file_path+'array_n_values/len_4_values.pkl')
    len_5_vals = open_data(file_path+'array_n_values/len_5_values.pkl')
    len_6_vals = open_data(file_path+'array_n_values/len_6_values.pkl')
    len_7_vals = open_data(file_path+'array_n_values/len_7_values.pkl')
    if current_version == None:
        current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
    if depth == None:
        depth = current_version['minimax_level_beat']+1
    test_version_path = file_path+'c4netST_versions/'+test_version
    len_n_vals_lst = [len_4_vals,len_5_vals,len_6_vals,len_7_vals]
    vals = [1,2,3,4]
    vals2 = [1.4,2.1,2.8,3.5]
    mults1 = [(w,x,y) for w in vals for x in vals for y in vals if w in [1,2]]
    mults2 = [(w,x,y) for w in vals for x in vals for y in vals if w in [3,4]]
    mults3 = [(w,x,y) for w in vals2 for x in vals2 for y in vals2 if w in [1.4,2.1]]
    mults4 = [(w,x,y) for w in vals2 for x in vals2 for y in vals2 if w in [2.8,3.5]]
    inputs = [[True, len_n_vals_lst, mults1,depth,in_game_iterations,test_version_path],
              [True, len_n_vals_lst, mults2,depth,in_game_iterations,test_version_path],
              [True, len_n_vals_lst, mults3,depth,in_game_iterations,test_version_path],
              [True, len_n_vals_lst, mults4,depth,in_game_iterations,test_version_path],
              [False, len_n_vals_lst, mults1,depth,in_game_iterations,test_version_path],
              [False, len_n_vals_lst, mults2,depth,in_game_iterations,test_version_path],
              [False, len_n_vals_lst, mults3,depth,in_game_iterations,test_version_path],
              [False, len_n_vals_lst, mults4,depth,in_game_iterations,test_version_path]]
    with Pool(num_proc) as p:
        results_lst = p.starmap(multi_play_mini_only_result,inputs)
    win_percent = round(100*sum(results_lst)/256,2)
    return(win_percent,results_lst)


def multi_testnet_vs_current(in_game_iterations = 400,
                             current_version_path = None,
                             file_path = file_path):
    """For testing the test version vs the current version. For multiprocessing"""
    if current_version_path == None:
        current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
        version_path = file_path+'c4netST_versions/'+current_version['name']
    else:
        version_path = current_version_path
    inputs = [[version_path, True, in_game_iterations, 25, True],
              [version_path, True, in_game_iterations, 25, True],
              [version_path, True, in_game_iterations, 25, True],
              [version_path, True, in_game_iterations, 25, True],
              [version_path, False, in_game_iterations, 25, True],
              [version_path, False, in_game_iterations, 25, True],
              [version_path, False, in_game_iterations, 25, True],
              [version_path, False, in_game_iterations, 25, True]]
    with Pool(8) as p:
        results_lst = p.starmap(multi_two_versions_play_only_result,inputs)
    win_percent = sum(results_lst)/2
    return(win_percent,results_lst)



def vs_mini_or_self(vs_mini,
                    as_x,
                    len_n_vals_lst,
                    mult_lst,
                    depth,
                    in_game_iterations = 400,
                    version_path = file_path+'c4netST_versions/testversion',
                    add_noise = False):
    """For testing the test version vs the current version and minimax at the same time. For multiprocessing"""
    if vs_mini:
        num_won = multi_play_mini_only_result(as_x,
                                              len_n_vals_lst,
                                              mult_lst,
                                              depth,
                                              in_game_iterations,
                                              version_path)
    else:
        num_won = multi_two_versions_play_only_result(current_version_path = version_path,
                                                      current_version_as_x = as_x, 
                                                      iterations = in_game_iterations,
                                                      num_games = 25,
                                                      add_noise = add_noise)
    return num_won

def multi_test(in_game_iterations = 400,
               version_path = None,
               depth = None,
               file_path = file_path):
    """For testing the test version vs the current version and minimax at the same time. For multiprocessing"""
    clear_numba()
    test_version_path = file_path+'c4netST_versions/testversion'
    len_4_vals = open_data('array_n_values/len_4_values.pkl')
    len_5_vals = open_data('array_n_values/len_5_values.pkl')
    len_6_vals = open_data('array_n_values/len_6_values.pkl')
    len_7_vals = open_data('array_n_values/len_7_values.pkl')
    current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
    current_version_path = file_path+'c4netST_versions/'+current_version['name']
    depth = current_version['minimax_level_beat']+1
    print('\nBeginning Tests')
    print('current version:',current_version['name'])
    print('Testing depth:',depth)
    len_n_vals_lst = [len_4_vals,len_5_vals,len_6_vals,len_7_vals]
    vals = [1,2,3,4]
    mults1 = [(w,x,y) for w in vals for x in vals for y in vals if w in [1,2]]
    mults2 = [(w,x,y) for w in vals for x in vals for y in vals if w in [3,4]]
    #       vs_mini, as_x, list_n_vals, mult_lsts,depth,in_game_iterations, current_version_path
    inputs = [[True, True, len_n_vals_lst, mults1, depth, in_game_iterations, test_version_path, False],
              [True, True, len_n_vals_lst, mults2, depth, in_game_iterations, test_version_path, False],
              [True, False, len_n_vals_lst, mults1, depth, in_game_iterations, test_version_path, False],
              [True, False, len_n_vals_lst, mults2, depth, in_game_iterations, test_version_path, False],
              [False, True, None, None, None, in_game_iterations, current_version_path, False],
              [False, False, None, None, None, in_game_iterations, current_version_path, False],
              [False, True, None, None, None, in_game_iterations, current_version_path, False],
              [False, False, None, None, None, in_game_iterations, current_version_path, False]]
    with Pool(8) as p:
        results_lst = p.starmap(vs_mini_or_self,inputs)
    win_percent_vs_mini = round(100*sum(results_lst[:4])/128,2)
    win_percent_vs_current = sum(results_lst[4:])
    return(win_percent_vs_mini, win_percent_vs_current, results_lst)

def watch_games_vs_mini_multi_depths(version_path,
                                     min_depth,
                                     max_depth,
                                     in_game_iterations,
                                     only_final_board = True,
                                     file_path = file_path):
    """For watching games between a specified version and a set of minimax algs.  
    To check that everything is working correctly"""
    clear_numba()
    len_4_vals = open_data(file_path+'array_n_values/len_4_values.pkl')
    len_5_vals = open_data(file_path+'array_n_values/len_5_values.pkl')
    len_6_vals = open_data(file_path+'array_n_values/len_6_values.pkl')
    len_7_vals = open_data(file_path+'array_n_values/len_7_values.pkl')
    len_n_vals_lst = [len_4_vals,len_5_vals,len_6_vals,len_7_vals]
    vals = [1,2,3,4]
    mults = [(w,x,y) for w in vals for x in vals for y in vals]
    depth_results = []
    for depth in range(min_depth,max_depth+1):
        results_as_x = 0
        results_as_o = 0
        results = 0
        games_played = 0
        draws = 0
        draws_as_x = 0
        draws_as_o = 0
        games_as_x = 0
        games_as_o = 0
        total_time = 0
        np.random.shuffle(mults)
        toc = tt()
        
        for mult in mults:
            if games_played >= 50:
                break
            print('\n------------------------------------------------\n')
            print('c4net was X')
            toc1=tt()
            result = watch_games_vs_mini(as_x=True,
                                         len_n_vals_lst=len_n_vals_lst,
                                         mult=mult,
                                         depth = depth,
                                         in_game_iterations = in_game_iterations,
                                         version_path=version_path,
                                         watch_full_games = not only_final_board,
                                         print_final_board = only_final_board)
            tic1 = tt()
            total_time = tic1 - toc
            if result == 2:
                draws += 1
                draws_as_x += 1
            else:
                results_as_x += result
                results += result
            games_as_x += 1
            games_played += 1
            print('Mini Depth:',depth)
            print('| row mult:', mult[0],'| col mult:', mult[1],'| diag mult:', mult[2],'|')
            print('\nPercent won:','{}%'.format(round(100*results/games_played,2)),'({}/{})'.format(results,games_played))
            print('Percent draws:','{}%'.format(round(100*draws/games_played,2)),'({}/{})'.format(results,games_played))
            print('\nPercent won as X:','{}%'.format(round(100*results_as_x/games_as_x,2)),'({}/{})'.format(results_as_x,games_as_x))
            print('Percent draws as X:','{}%'.format(round(100*draws_as_x/games_as_x,2)),'({}/{})'.format(results_as_x,games_as_x))
            if games_as_o>0:
                print('\nPercent won as O:','{}%'.format(round(100*results_as_o/games_as_o,2)),'({}/{})'.format(results_as_o,games_as_o))
                print('Percent draws as O:','{}%'.format(round(100*draws_as_o/games_as_o,2)),'({}/{})'.format(results_as_o,games_as_o))
            print('\nGame time:',round(tic1-toc1,2))
            print('Avg game time:',round(total_time/games_played,2))
            print('\n------------------------------------------------\n')
            print('c4net was O')
            toc1=tt()
            result = watch_games_vs_mini(as_x=False,
                                         len_n_vals_lst=len_n_vals_lst,
                                         mult=mult,
                                         depth = depth,
                                         in_game_iterations = in_game_iterations,
                                         version_path=version_path,
                                         watch_full_games = not only_final_board,
                                         print_final_board = only_final_board)
            tic1 = tt()
            total_time = tic1 - toc
            if result == 2:
                draws += 1
                draws_as_o += 1
            else:
                results_as_o += result
                results += result
            games_as_o+=1
            games_played+=1
            
            print('Mini Depth:',depth)
            print('| row mult:', mult[0],'| col mult:', mult[1],'| diag mult:', mult[2],'|')
            print('\nPercent won:','{}%'.format(round(100*results/games_played,2)),'({}/{})'.format(results,games_played))
            print('Percent draws:','{}%'.format(round(100*draws/games_played,2)),'({}/{})'.format(results,games_played))
            print('\nPercent won as X:','{}%'.format(round(100*results_as_x/games_as_x,2)),'({}/{})'.format(results_as_x,games_as_x))
            print('Percent draws as X:','{}%'.format(round(100*draws_as_x/games_as_x,2)),'({}/{})'.format(results_as_x,games_as_x))
            print('\nPercent won as O:','{}%'.format(round(100*results_as_o/games_as_o,2)),'({}/{})'.format(results_as_o,games_as_o))
            print('Percent draws as O:','{}%'.format(round(100*draws_as_o/games_as_o,2)),'({}/{})'.format(results_as_o,games_as_o))
            print('\nGame time:',round(tic1-toc1,2))
            print('Avg game time:',round(total_time/games_played,2))
        
        depth_results.append({'depth':depth,
                              'percent_won':round(100*results/games_played,2),
                              'percent_draws':round(100*draws/games_played,2),
                              'percent_won_as_x':round(100*results_as_x/games_as_x,2),
                              'percent_draws_as_x':round(100*draws_as_x/games_as_x,2),
                              'percent_won_as_0':round(100*results_as_o/games_as_o,2),
                              'percent_draws_as_o':round(100*draws_as_o/games_as_o,2)})
    return(depth_results)
# multi_play_mini_only_result, multi_two_versions_play_only_result

def run_many_depth_tests(min_depth = 1, 
                         max_depth =5,
                         test_version_path = None):
    """For watching games between a specified version and a set of minimax algs.  
    To check that everything is working correctly"""
    results = []
    for n in range(min_depth,max_depth+1):
        result = get_win_percent_testnet_vs_mini(training_iterations = 1,
                                                 in_game_iterations=400,
                                                 num_games = 20,
                                                 print_true = True,
                                                 depth = n,
                                                 test_version_path = test_version_path)
        results.append(result)
    return results


# run_many_depth_tests(1,5,'C:/Users/matth/python/connectfour/c4netST_versions/c4netST_5-0')



def make_and_fit_new_model(batch_size=10,
                           epochs=5,
                           current_version = None,
                           file_path = file_path):
    """Makes and fits a new model to the data"""
    from make_keras_models import fit_model,make_model
    if current_version == None:
        current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
    data_version = '{}_{}_{}'.format(current_version['minimax_level_beat'],
                                     current_version['mini_level_version'],
                                     current_version['training_cycles'])
    data_name_self = file_path+'c4netST_data/{}_{}_{}.pkl'.format('self','play',data_version)
    data_self = mcts_data(data_name_self)
    data_self = reflect_and_concat_data(data_self)
    c4netST = make_model()
    if current_version['name'] == 'c4netST_0-0' or current_version['training_cycles'] == 0:
        recent_data = data_self
    else:
        recent_data = open_data(file_path+'c4netST_data/recent_data.pkl')
        recent_data = [np.concatenate((data_self[0],recent_data[0])),
                       np.concatenate((data_self[1],recent_data[1])),
                       np.concatenate((data_self[2],recent_data[2]))]
    boards,scores,results = shuffle_mcts_data(recent_data)
    train_x = boards
    train_y_results =  results
    train_y_scores =  scores
    train_y = [train_y_scores,train_y_results]
    history = fit_model(model=c4netST,
                        train_x = train_x,
                        train_y=train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_percent=.05)
    print_last_epoch_history(history)
    test_version_path = file_path+'c4netST_versions/testversion'
    c4netST.save(test_version_path)
    gameboards_trained_on = len(train_x)
    del c4netST
    import tensorflow.keras.backend as K
    K.clear_session()
    return(gameboards_trained_on)



def make_and_fit_transfer_model(batch_size=1024,
                                epochs=30,
                                version = None,
                                file_path = file_path):
    """Makes and fits a transfer model to the data"""
    from make_keras_models import fit_model,transfer_model
    c4netST = transfer_model(file_path+'/c4netST_versions/testversion')
    if version == None:
        current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
        data_version = '{}_{}_{}'.format(current_version['minimax_level_beat'],
                                         current_version['mini_level_version'],
                                         current_version['training_cycles'])
        data_name_self = file_path+'c4netST_data/{}_{}_{}.pkl'.format('self','play',data_version)
        data_self = mcts_data(data_name_self)
        data_self = reflect_and_concat_data(data_self)
        if current_version['name'] == 'c4netST_0-0' or current_version['training_cycles'] == 0:
            recent_data = data_self
        else:
            recent_data = open_data(file_path+'c4netST_data/recent_data.pkl')
            recent_data = [np.concatenate((data_self[0],recent_data[0])),
                           np.concatenate((data_self[1],recent_data[1])),
                           np.concatenate((data_self[2],recent_data[2]))]
    else:
        recent_data = open_data(file_path+'c4netST_data/recent_data.pkl')
    boards,scores,results = shuffle_mcts_data(recent_data)
    train_x = boards
    train_y_results =  results
    train_y_scores =  scores
    train_y = [train_y_scores,train_y_results]
    history = fit_model(model=c4netST,
                        train_x = train_x,
                        train_y=train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_percent=.05)
    
    test_version_path = file_path+'c4netST_versions/testversion'
    c4netST.save(test_version_path)
    print_last_epoch_history(history)
    import tensorflow.keras.backend as K
    K.clear_session()

def make_and_fit_new_model_specific_data(batch_size=1024,
                                         epochs=30,
                                         data_level = 'level_0_data.pkl',
                                         test_version = 'testversion',
                                         file_path = file_path):
    """Makes and fits a new model to a specific set of data,
    for testing different models on premade data"""
    data_path = file_path+'c4netST_data/'+data_level
    test_version_path = file_path+'c4netST_versions/'+test_version
    from make_keras_models import fit_model,make_model_3
    c4netST = make_model_3()
    recent_data = open_data(data_path)
    boards,scores,results = shuffle_mcts_data(recent_data)
    train_x = boards
    train_y_results =  results
    train_y_scores =  scores
    train_y = [train_y_scores,train_y_results]
    history = fit_model(model=c4netST,
                        train_x = train_x,
                        train_y=train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_percent=.05)
    print_last_epoch_history(history)
    c4netST.save(test_version_path)
    gameboards_trained_on = len(train_x)
    del c4netST
    import tensorflow.keras.backend as K
    K.clear_session()
    return(gameboards_trained_on)



def make_and_fit_transfer_model_specific_data(batch_size=1024,
                                              epochs=30,
                                              data_level = 'level_1_data.pkl',
                                              test_version = 'testversion',
                                              file_path = file_path):
    """Makes and fits a transfer model to a specific set of data,
    for testing different models on premade data"""
    # data_path = file_path+'c4netST_data/'+data_level
    test_version_path = file_path+'c4netST_versions/'+test_version
    from make_keras_models import fit_model,transfer_model_3
    c4netST = transfer_model_3(test_version_path)
    recent_data = open_data(file_path+'c4netST_data/recent_data.pkl')
    boards,scores,results = shuffle_mcts_data(recent_data)
    train_x = boards
    train_y_results =  results
    train_y_scores =  scores
    train_y = [train_y_scores,train_y_results]
    history = fit_model(model=c4netST,
                        train_x = train_x,
                        train_y=train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_percent=.05)
    c4netST.save(test_version_path)
    print_last_epoch_history(history)
    import tensorflow.keras.backend as K
    K.clear_session()


def update_version_and_data(vs_current_percent,
                            vs_mini_percent,
                            gameboards_trained_on=300000,
                            results_lst_mini = [],
                            results_lst_current=[],
                            file_path = file_path):
    """Saves everything, and updates file location based on test results"""
    from tensorflow.keras.models import load_model
    versions = open_data(file_path+'c4netST_versions/versions.pkl')
    current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
    version_path = file_path+'c4netST_versions/'+current_version['name']
    if vs_mini_percent >=80:
        c4netST = load_model(file_path+'c4netST_versions/testversion')
        current_version['results_lst_vs_mini'] = results_lst_mini
        current_version['results_lst_vs_current'] = results_lst_current
        current_version['minimax_level_beat']+=1
        current_version['mini_level_version']=0
        current_version['training_cycles']=0
        current_version['games_trained_on']=gameboards_trained_on
        current_version['name'] = 'c4netST'+'_{}-{}'.format(current_version['minimax_level_beat'],current_version['mini_level_version'])
        current_version['win_percent_vs_mini'] = vs_mini_percent
        current_version['win_percent_vs_previous'] = vs_current_percent
        current_version['win_percent_vs_current'] = None
        version_path = file_path+'c4netST_versions/'+current_version['name']
        c4netST.save(version_path)
        del c4netST
        # make_recent_data(minimax_level_beat = current_version['minimax_level_beat'], versions = versions )
    elif vs_current_percent >= 55:
        c4netST = load_model(file_path+'c4netST_versions/testversion')
        current_version['results_lst_vs_mini'] = results_lst_mini
        current_version['results_lst_vs_current'] = results_lst_current
        current_version['mini_level_version'] += 1
        current_version['training_cycles'] = 0
        current_version['win_percent_vs_mini'] = vs_mini_percent
        current_version['win_percent_vs_previous'] = vs_current_percent
        current_version['win_percent_vs_current'] = None
        current_version['games_trained_on']=gameboards_trained_on
        current_version['name'] = 'c4netST'+'_{}-{}'.format(current_version['minimax_level_beat'],current_version['mini_level_version'])
        version_path = file_path+'c4netST_versions/'+current_version['name']
        c4netST.save(version_path)
        make_recent_data()
        del c4netST
    else:
        current_version['results_lst_vs_mini'] = results_lst_mini
        current_version['results_lst_vs_current'] = results_lst_current
        current_version['win_percent_vs_mini'] = vs_mini_percent
        current_version['win_percent_vs_current'] = vs_current_percent
        current_version['training_cycles'] += 1
        make_recent_data()
    versions.append(current_version)
    save_data(current_version,file_path+'c4netST_versions/current_version.pkl')
    save_data(versions,file_path+'c4netST_versions/versions.pkl')
    import tensorflow.keras.backend as K
    K.clear_session()

def print_last_epoch_history(history):
    """prints the info from the final training epoch"""
    history_dict = history.history
    keys = ['loss',
            'score_output_loss', 
            'result_output_loss', 
            'score_output_categorical_crossentropy', 
            'score_output_categorical_accuracy',
            'result_output_binary_crossentropy',
            'result_output_binary_accuracy']
    for key in keys:
        print('')
        print('-'*50,'\n')
        print('    '+key+':',round(history_dict[key][-1],4))
        print('val_'+key+':',round(history_dict['val_'+key][-1],4))
    print('-'*50,'\n')

def make_recent_data(minimax_level_beat=None,
                      versions = None,
                      file_path = file_path):
    """puts all of the recently generated data as one file"""
    if minimax_level_beat == None:
        current_version = open_data(file_path+'c4netST_versions/current_version.pkl')
        minimax_level_beat = current_version['minimax_level_beat']
        mini_level_version = current_version['mini_level_version']
    if versions == None:
        versions = open_data(file_path+'c4netST_versions/versions.pkl')
    num_added = 0
    for version in versions:
        if version['minimax_level_beat'] == minimax_level_beat and version['mini_level_version'] == mini_level_version:
            data_version = '{}_{}_{}'.format(version['minimax_level_beat'],
                                             version['mini_level_version'],
                                             version['training_cycles'])
            data_name_self = file_path+'c4netST_data/{}_{}_{}.pkl'.format('self','play',data_version)
            data_self = mcts_data(data_name_self)
            data_self = reflect_and_concat_data(data_self)
            if num_added == 0:
                data = data_self
            else:
                data = [np.concatenate((data[0],data_self[0])),
                        np.concatenate((data[1],data_self[1])),
                        np.concatenate((data[2],data_self[2]))]
            num_added+=1
    
    save_data(data,file_path+'c4netST_data/recent_data.pkl')


def make_level_data(level,
                    versions = None,
                    file_path = file_path):
    """puts all of the recently generated data as one file for a specific minimax level beat"""
    if versions == None:
        versions = open_data(file_path+'c4netST_versions/versions.pkl')
    num_added = 0
    for n in range(1):
        for version in versions:
            if version['minimax_level_beat'] == level:
                data_version = '{}_{}_{}'.format(version['minimax_level_beat'],
                                                 version['mini_level_version'],
                                                 version['training_cycles'])
                data_name_self = file_path+'c4netST_data/{}_{}_{}.pkl'.format('self','play',data_version)
                data_self = mcts_data(data_name_self)
                data_self = reflect_and_concat_data(data_self)
                if num_added == 0:
                    data = data_self
                else:
                    data = [np.concatenate((data[0],data_self[0])),
                            np.concatenate((data[1],data_self[1])),
                            np.concatenate((data[2],data_self[2]))]
                num_added+=1
    file_name = 'level_'+str(level)+'_data.pkl'
    save_data(data,file_path+'c4netST_data/'+file_name)

def reflect_and_concat_data(mcts_data):
    """the board is symmetric, so we can double our data by reflection"""
    reflected_data = [np.flip(mcts_data[0],axis=1),np.flip(mcts_data[1],axis=1),mcts_data[2]]
    data = [np.concatenate((mcts_data[0],reflected_data[0])),
            np.concatenate((mcts_data[1],reflected_data[1])),
            np.concatenate((mcts_data[2],reflected_data[2]))]
    return(data)


def make_initial_everything(file_path = file_path):
    """initializes everything for a new model to be trained.  
    You should move all of the old data before you run this, 
    if you wish to keep the old data."""
    from minimax_functions import new_board
    initial_version = {'name':'c4netST_0-0',
                        'minimax_level_beat':0,
                        'mini_level_version':0,
                        'training_cycles':0,
                        'games_trained_on':0,
                        'win_percent_vs_mini':0,
                        'win_percent_vs_current':0,
                        'win_percent_vs_previous':0}
    from Using_Keras_current import fit_model,make_model
    versions = [initial_version]
    save_data(initial_version,file_path+'c4netST_versions/current_version.pkl')
    save_data(versions,file_path+'c4netST_versions/versions.pkl')
    fake_boards = np.array([np.moveaxis(np.array([new_board(),new_board()]),0,-1),
                            np.moveaxis(np.array([new_board(),new_board()]),0,-1)])
    fake_results = np.array([1,1])
    fake_scores = np.ones((2,7))/7
    train_x = fake_boards
    train_y = [fake_scores,fake_results]
    initial_model = make_model()
    fit_model(model=initial_model,
              train_x = train_x,
              train_y=train_y,
              batch_size=1,
              epochs=1,
              validation_percent=.1)
    initial_version_path = file_path+'c4netST_versions/c4netST_0-0'
    test_version_path = file_path+'c4netST_versions/testversion'
    initial_model.save(initial_version_path)
    initial_model.save(test_version_path)




# for testing! 
# def test_pipeline_loop():
#     #save real versions
#     current_version = open_data('c4netST_versions/current_version.pkl')
#     save_data(current_version,'c4netST_versions/real_current_version.pkl')
#     versions = open_data('c4netST_versions/versions.pkl')
#     save_data(versions,'c4netST_versions/real_versions.pkl')
#     current_version = {'name':current_version['name'],
#                         'minimax_level_beat':4,
#                         'mini_level_version':111,
#                         'training_cycles':111,
#                         'games_trained_on':2000,
#                         'current_win_percent':0}
#     c4netST_training_pipeline(training_loops=1,
#                               num_hours=.0000001,
#                               train_for_time = False,
#                               mini_games=1,
#                               in_game_iterations=1,
#                               training_iterations=1,
#                               num_self_games_per_tree=1,
#                               num_trees=1,
#                               batch_size = 10000000000000000,
#                               epochs = 1,
#                               print_true =True,
#                               current_version = current_version,
#                               num_test_games = 1)
#     current_version = open_data('c4netST_versions/real_current_version.pkl')
#     save_data(current_version,'c4netST_versions/current_version.pkl')
#     versions = open_data('c4netST_versions/real_versions.pkl')
#     save_data(versions,'c4netST_versions/versions.pkl')
#     make_recent_data(minimax_level_beat=current_version['minimax_level_beat'],
#                       versions = versions)






