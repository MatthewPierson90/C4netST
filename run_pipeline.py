# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:25:23 2021

@author: Matthew
"""
from c4netST_training_pipeline_current import make_and_fit_transfer_model, update_version_and_data, tt, watch_games_vs_mini_multi_depths 
from c4netST_training_pipeline_current import multi_testnet_vs_current, multi_testnet_vs_mini, c4netST_training_pipeline


if __name__=='__main__':
    c4netST_training_pipeline(training_loops = 4,
                              in_game_itterations_vs_self = 1000,
                              training_itterations = 1,
                              num_self_games_per_tree = 250,
                              num_trees = 18,
                              batch_size = 1024,
                              epochs = 20,
                              print_true = True,
                              use_multi_self = True, #if True do behind if __name__ == __main__ 
                              use_multi_test = True,
                              num_proc_multi_self = 6)
    # results = watch_games_vs_mini_multi_depths(version_path = 'C:/Users/matth/python/connectfour/c4netST/V2/c4netST_versions/c4netST_0-0',
    #                                            min_depth = 1,
    #                                            max_depth = 4,
    #                                            in_game_itterations = 1000,
    #                                            only_final_board = True)
    # # update_time = 0
    # testing_time = 0
    # training_time = 0
    # print_true=False
    # training_itterations = 1
    # epochs = 15
    # batch_size = 1024
    # in_game_itterations_vs_mini = 1000
    # in_game_itterations_vs_self = 1000
    # times_through = 0
    # vs_mini_percent = 0
    # vs_current_percent = 0
    # while vs_mini_percent < 60 and vs_current_percent < 55 and times_through < 5:
    #     training_toc = tt()
    #     print('Times Through:',times_through)
    #     gameboards_trained_on = make_and_fit_transfer_model(epochs = epochs, 
    #                                                         batch_size = batch_size,
    #                                                         version = versions[-2])
    #     training_tic = tt()
    #     training_time += training_tic-training_toc
    #     testing_toc = tt()
    #     # vs_mini_percent, vs_current_percent, results_lst= multi_test(in_game_itterations = in_game_itterations_vs_self)
    #     vs_mini_percent, results_lst_mini = multi_testnet_vs_mini(in_game_itterations = in_game_itterations_vs_mini)
    #     print('Percent won vs mini:',vs_mini_percent)
    #     print('Mini Results lst (32 games each):',results_lst_mini)
    #     print('Percent won as X:',round(100*sum(results_lst_mini[:4])/128,2))
    #     print('Percent won as O:',round(100*sum(results_lst_mini[4:])/128,2))
    #     vs_current_percent, results_lst_current = multi_testnet_vs_current(in_game_itterations = in_game_itterations_vs_self)
    #     print('Percent won vs current:',vs_current_percent)
    #     print('Results vs current version (25 games each):',results_lst_current)
    #     print('Percent won as X:',sum(results_lst_current[:4]))
    #     print('Percent won as O:',sum(results_lst_current[4:]))
    #     testing_tic = tt()
    #     testing_time += testing_tic-testing_toc
    #     times_through+=1
    # update_toc = tt()
    # update_version_and_data(vs_current_percent = vs_current_percent,
    #                         vs_mini_percent = vs_mini_percent,
    #                         gameboards_trained_on = gameboards_trained_on,
    #                         results_lst_mini = results_lst_mini,
    #                         results_lst_current = results_lst_current)
    # update_tic = tt()
    # update_time += update_tic-update_toc
    # import tensorflow.keras.backend as K
    # K.clear_session()