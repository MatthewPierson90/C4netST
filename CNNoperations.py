# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:40:40 2021

@author: Matthew
"""


from numba import njit
import numba as nb
import numpy as np
import copy


def print_lst(lst):
    print('#'*75)
    for n in rlen(lst):
        print(lst[n])
        if n == len(lst)-1:
            print('#'*75)
        else:
            print('-'*50)
    print('\n')

def rlen(lst,start =0):
    return range(start,len(lst))

################################################## Cleaning Data


@njit(nogil=True, cache=True)
def make_conv_array(block_array,filter_height,filter_width,input_height,input_width):
    #new_blocks_are_square new block has dims input_height-sub_block_size by input_width-sub_block_size
    new_mat = np.zeros((filter_height*filter_width,
                        (input_width+1-filter_width)*(input_height+1-filter_height)))
    count1=0
    for j in range(input_height+1-filter_height):
        for i in range(input_width+1-filter_width):
            filter_array=np.zeros(filter_width*filter_height)
            count2 = 0
            for k in range(filter_height):
                for r in range(filter_width):
                    s = j+i+j*(input_width-1)
                    
                    filter_array[count2]= block_array[s+r+input_width*k]
                    count2+=1

            new_mat[:,count1] = filter_array
            count1+=1
    return(new_mat)



@njit(nogil=True, cache=True)
def make_conv_array_2d(block_array,filter_height,filter_width,input_height,input_width):
    conv_inputs = []
    for n in range(len(block_array)):
            conv_inputs.append(make_conv_array(block_array[n],
                                               filter_height,
                                               filter_width,
                                               input_height,
                                               input_width))
    return conv_inputs


@njit(nogil=True, cache=True)
def make_conv_array_3d(block_array,filter_height,filter_width,input_height,input_width):
    conv_inputs = []
    for n in range(len(block_array)):
            for m in range(len(block_array[n])):
                conv_inputs.append(make_conv_array(block_array[n][m],
                                                   filter_height,
                                                   filter_width,
                                                   input_height,
                                                   input_width))
    # conv_inputs = np.array(conv_inputs)
    return conv_inputs


@njit(nogil=True, cache=True)
def make_square_conv_array(block_array,filter_dim,input_height,input_width):
    return make_conv_array(block_array,
                           filter_dim,
                           filter_dim,
                           input_height,
                           input_width)


@njit(nogil=True, cache=True)
def make_full_conv_array(block_array,filter_height, filter_width,oldH,oldW):
    newH = oldH+2*filter_height-2
    newW = oldW+2*filter_width-2
    new_array = [ 0. for n in range(newW*(filter_height-1)+(filter_width-1))]
    count=0
    for n in range(len(block_array)):
        if count == oldW:
            new_array+=[0. for n in range(2*filter_width-2)]
            count=0
        new_array+=[block_array[n]]
        count+=1
    new_array+=[ 0. for n in range(newW*(filter_height-1)+(filter_width-1))]
    new_array = np.array(new_array)
    to_return = make_conv_array(new_array,filter_height,filter_width,newH,newW)
    # print(type(to_return))
    # print(to_return)
    return(to_return)

@njit(nogil=True, cache=True)
def make_full_conv_array2d(block_array,filter_height, filter_width,input_height,input_width):
    conv_inputs = []
    for n in range(len(block_array)):
        to_make = block_array[n]
        # print(to_make.shape)
        to_append = make_full_conv_array(to_make,
                                         filter_height,
                                         filter_width,
                                         input_height,
                                         input_width)
        conv_inputs.append(to_append)
    return conv_inputs


def add_zeros(old_array, filter_height, filter_width):
    old_height= old_array.shape[0]
    old_width = old_array.shape[1]
    new_height = old_height+2*filter_height-2
    new_width = old_width+2*filter_width-2
    new_array = np.zeros((new_height,new_width))
    new_array[filter_height-1:filter_height-1+old_height,filter_width-1:filter_width-1+old_width]=old_array
    return(new_array)

def add_zeros_to_all(old_array, filter_height, filter_width):
    num = old_array.shape[0]
    old_height= old_array.shape[1]
    old_width = old_array.shape[2]
    new_height = old_height+2*filter_height-2
    new_width = old_width+2*filter_width-2
    new_array = np.zeros((num,new_height,new_width))
    new_array[:,filter_height-1:filter_height-1+old_height,filter_width-1:filter_width-1+old_width]=old_array
    return(new_array)

################################################## backprop functions




def conv_mult(input_matrix,filter_matrix):
    if filter_matrix.ndim ==2:
        filter_height = filter_matrix.shape[0]
        filter_width = filter_matrix.shape[1]
    elif filter_matrix.ndim ==3:
        filter_height = filter_matrix.shape[1]
        filter_width = filter_matrix.shape[2]
    if input_matrix.ndim ==2:
        input_height = input_matrix.shape[0]
        input_width = input_matrix.shape[1]
        output_height = input_height-filter_height+1
        output_width = input_width-filter_width+1
        output_matrix = np.zeros((output_height,output_width))
        for m in range(output_height):
            for n in range(output_width):
                output_matrix[m][n] = (input_matrix[m:m+filter_height,n:n+filter_width]*filter_matrix).sum()
        return(output_matrix)
    if input_matrix.ndim == 3:
        num_outputs = input_matrix.shape[0]
        input_height = input_matrix.shape[1]
        input_width = input_matrix.shape[2]
        output_height = input_height-filter_height+1
        output_width = input_width-filter_width+1
        output_matrix = np.zeros((num_outputs,output_height,output_width))
        for m in range(output_height):
            for n in range(output_width):
                output_matrix[:,m,n] = (input_matrix[:,m:m+filter_height,n:n+filter_width]*filter_matrix).sum(axis=(1,2))
        return(output_matrix)




def ssigmoid(z):
    return(2*sigmoid(.5*z)-1)

def ssigmoidprime(z):
    return 4*sigmoid(.5*z)*(1-sigmoid(.5*z))


def scale_scores(z):
    large = 10
    slope = 1/large
    new = copy.deepcopy(z)
    new = slope*new
    new[new<=-1]=-1
    new[new>=1]=1
    return new


def relu(z):
    new = copy.deepcopy(z)
    new[new<0]=.001*new[new<0]
    return new


def reluprime(z):
    new = copy.deepcopy(z)
    new[new<0]=.001
    new[new>0]=1
    return new

def sigmoid(z):
    return(1/(1+np.exp(-z)))

def siginv(y):
    return(np.log(y)-np.log(1-y))


def sigmoidprime(z):
    return sigmoid(z)*(1-sigmoid(z))

def mm(M,x):
    return(np.dot(M,x))


def dcost(a,y):
    return 2*(a-y)


def calc_cost(a,y):
    m = y.shape[1]
    c = ((a-y)**2/m).sum()
    return(c)
