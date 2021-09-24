# C4netST (self trained)
Inspired by Alpha Zero.  

This program runs a Monte Carlo Tree Search (MCTS) algorithm for playing connect four.  
To make each move, makes 1000 simulated moves and greedily chooses the move with the highest number of posiitive outcomes.  
The simulated moves are greedily chosen by using a combination of previous data from the current simulation and the output of a neural net which predicts which move is the best current move. The neural net is trained on data generated by previous iterations of the tree playing itself, for each board state it predicts the best move for the current player and the current player's chance of winning.  

The training process is as follows:

The current iteration of C4neST plays against itself to generate data.  After a set number of games, usually 1000, new testnets are trained and tested against the current version as well as against a set of minimax algorithms.  If the testnet wins 55% of the games vs the current C4netST, or by beating the minimax algorithms by a rate of 85% or better, then the testnet is set as the current version and becomes the data generator.   

I have been running the data generation and testing games in parallel on the CPU, usually 8 games at a time, the neural net is fit on a GPU.

Package versions I have been using:

numba 0.53.1
numpy 1.20.3
python 3.9.6
tensorflow 2.5.0

