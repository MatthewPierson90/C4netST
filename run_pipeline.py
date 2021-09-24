"""
@author: Matthew

runs the pipeline
"""
from c4netST_training_pipeline import c4netST_training_pipeline
# from c4netST_training_pipeline_current import *

if __name__=='__main__':
    c4netST_training_pipeline(training_loops = 1,
                              in_game_itterations_vs_self = 1000,
                              training_itterations = 1,
                              num_self_games_per_tree = 10,
                              num_trees = 6,
                              batch_size = 1024,
                              epochs = 15,
                              print_true = True,
                              use_multi_self = True, #if True do behind if __name__ == __main__ 
                              use_multi_test = True,
                              num_proc_multi_self = 6)
