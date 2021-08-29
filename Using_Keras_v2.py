# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 06:27:05 2021

@author: Matthew
"""
# needs to be run in the tf-gpu environment


# trying V2 but for scores
def rlen(lst,start =0):
    return range(start,len(lst))

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(50)
from tensorflow.keras import layers

import numpy as np

import time
tt = time.time


def soft(array):
    soft_inside = layers.Softmax(axis=0)
    return(soft_inside(array))
tt = time.time

def sigmoid(z):
    return(1/(1+np.exp(-z/20)))


# make the model
def make_model_2():
    import os
    from logging import getLogger
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    getLogger('tensorflow').setLevel(50)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    # Define Layers
    inputs = keras.Input(shape=(7,6,2), name='input')
    
    #shared layers
    layer1_square = layers.Conv2D(20, 5, activation='relu', name='square_conv1')
    layer1_row = layers.Conv2D(10, (7,1), activation='relu', name='row_conv')
    layer1_col = layers.Conv2D(10, (1,6), activation='relu', name='col_conv')
    layer2_square = layers.Conv2D(20, 2, activation='relu', name='square_conv2')
    layer2_row = layers.Conv2D(10, (3,1), activation='relu', name='row_conv2')
    layer2_col = layers.Conv2D(10, (1,2), activation='relu', name='col_conv2')
    layer2_square_flatten = layers.Flatten()
    layer2_square_flatten = layers.Flatten()
    layer2_row_flatten = layers.Flatten()
    layer2_col_flatten = layers.Flatten()
    layer1_row_flatten = layers.Flatten()
    layer1_col_flatten = layers.Flatten()
    
    
    #score layers:
    
    slayer3_dense = layers.Dense(100, activation = 'relu', name = 'sdense_layer3')
    slayer4_dense = layers.Dense(30, activation = 'relu', name = 'sdense_layer4')
    soutput_layer = layers.Dense(7,activation = 'softmax', name = 'score_output')

    #result layers
    rlayer3_dense = layers.Dense(100, activation = 'relu', name = 'rdense_layer3')
    rlayer4_dense = layers.Dense(30, activation = 'relu', name = 'rdense_layer4')
    routput_layer = layers.Dense(1,activation = 'sigmoid', name = 'result_output')
    # make graph
    
    # layer 1 ----------------------------------------
    x = layer1_square(inputs)
    y = layer1_row(inputs)
    z = layer1_col(inputs)

    # layer 2 -----------------------------------------
    
    x1 = layer2_square(x)
    x2 = layer2_row(x)
    x3 = layer2_col(x)
    
    
    # Flatten Layer 2
    
    x1 = layer2_square_flatten(x1)
    x2 = layer2_row_flatten(x2)
    x3 = layer2_col_flatten(x3)
    y = layer1_row_flatten(y)
    z = layer1_col_flatten(z)
    
    x = layers.Concatenate()([x1,x2,x3,y,z])
    # layer 3 ---------------------------------------
    
    sx = slayer3_dense(x) 
    rx = rlayer3_dense(x) 
    
    # layer 4 ---------------------------------------
    sx = slayer4_dense(sx) 
    rx = rlayer4_dense(rx)
    # output
    
    soutputs = soutput_layer(sx)
    routputs = routput_layer(rx)
    
    model = keras.Model(inputs = inputs, outputs = [soutputs,routputs], name = 'c4net')
    
    return(model)

def make_model():
    import os
    from logging import getLogger
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    getLogger('tensorflow').setLevel(50)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    # Define Layers
    inputs = keras.Input(shape=(7,6,2), name='input')
    
    #score layers:
    slayer1_square = layers.Conv2D(20, 5, activation='relu', name='ssquare_conv1')
    slayer1_row = layers.Conv2D(10, (7,1), activation='relu', name='srow_conv')
    slayer1_col = layers.Conv2D(10, (1,6), activation='relu', name='scol_conv')
    slayer2_square = layers.Conv2D(20, 2, activation='relu', name='ssquare_conv2')
    slayer2_row = layers.Conv2D(10, (3,1), activation='relu', name='srow_conv2')
    slayer2_col = layers.Conv2D(10, (1,2), activation='relu', name='scol_conv2')
    slayer2_square_flatten = layers.Flatten()
    slayer2_square_flatten = layers.Flatten()
    slayer2_row_flatten = layers.Flatten()
    slayer2_col_flatten = layers.Flatten()
    slayer1_row_flatten = layers.Flatten()
    slayer1_col_flatten = layers.Flatten()
    slayer3_dense = layers.Dense(100, activation = 'relu', name = 'sdense_layer')
    soutput_layer = layers.Dense(7,activation = 'softmax', name = 'score_output')

    #result layers
    rlayer1_square = layers.Conv2D(20, 5, activation='relu', name='rsquare_conv1')
    rlayer1_row = layers.Conv2D(10, (7,1), activation='relu', name='rrow_conv')
    rlayer1_col = layers.Conv2D(10, (1,6), activation='relu', name='rcol_conv')
    rlayer2_square = layers.Conv2D(20, 2, activation='relu', name='rsquare_conv2')
    rlayer2_row = layers.Conv2D(10, (3,1), activation='relu', name='rrow_conv2')
    rlayer2_col = layers.Conv2D(10, (1,2), activation='relu', name='rcol_conv2')
    rlayer2_square_flatten = layers.Flatten()
    rlayer2_square_flatten = layers.Flatten()
    rlayer2_row_flatten = layers.Flatten()
    rlayer2_col_flatten = layers.Flatten()
    rlayer1_row_flatten = layers.Flatten()
    rlayer1_col_flatten = layers.Flatten()
    rlayer3_dense = layers.Dense(100, activation = 'relu', name = 'rdense_layer')
    routput_layer = layers.Dense(1,activation = 'sigmoid', name = 'result_output')
    # make graph
    
    # layer 1 ----------------------------------------
    sx = slayer1_square(inputs)
    sy = slayer1_row(inputs)
    sz = slayer1_col(inputs)
    
    rx = rlayer1_square(inputs)
    ry = rlayer1_row(inputs)
    rz = rlayer1_col(inputs)
    # layer 2 -----------------------------------------
    
    sx1 = slayer2_square(sx)
    sx2 = slayer2_row(sx)
    sx3 = slayer2_col(sx)
    
    rx1 = rlayer2_square(rx)
    rx2 = rlayer2_row(rx)
    rx3 = rlayer2_col(rx)
    
    # Flatten Layer 2
    
    sx1 = slayer2_square_flatten(sx1)
    sx2 = slayer2_row_flatten(sx2)
    sx3 = slayer2_col_flatten(sx3)
    sy = slayer1_row_flatten(sy)
    sz = slayer1_col_flatten(sz)
    
    rx1 = rlayer2_square_flatten(rx1)
    rx2 = rlayer2_row_flatten(rx2)
    rx3 = rlayer2_col_flatten(rx3)
    ry = rlayer1_row_flatten(ry)
    rz = rlayer1_col_flatten(rz)
    
    sx = layers.Concatenate()([sx1,sx2,sx3,sy,sz])
    rx = layers.Concatenate()([rx1,rx2,rx3,ry,rz])
    # layer 3 ---------------------------------------
    
    sx = slayer3_dense(sx) 
    rx = rlayer3_dense(rx) 
    
    # output
    
    soutputs = soutput_layer(sx)
    routputs = routput_layer(rx)
    
    model = keras.Model(inputs = inputs, outputs = [soutputs,routputs], name = 'c4net')
    
    return(model)
# view the model
def view_model(model,make_plot=False,model_name='c4net'):
    import os
    from logging import getLogger
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    getLogger('tensorflow').setLevel(50)
    from tensorflow.keras import utils
    print(model.summary())
    if make_plot:
        save_as = model_name+'_shape.png'
        utils.plot_model(model, save_as, show_shapes=True)


def transfer_model(version_path):
    import os
    from logging import getLogger
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    getLogger('tensorflow').setLevel(50)
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import load_model
    model = load_model(version_path)
    inputs = keras.Input(shape=(220,))
    dense3_reset = layers.Dense(100,kernel_initializer= keras.initializers.he_normal)
    rout_reset = layers.Dense(1,kernel_initializer=keras.initializers.he_normal)
    sout_reset = layers.Dense(7,kernel_initializer=keras.initializers.he_normal)
    x = dense3_reset(inputs)
    x1 = rout_reset(x)
    x2 = sout_reset(x)
    model.get_layer(name = 'rdense_layer').set_weights(dense3_reset.get_weights())
    model.get_layer(name = 'sdense_layer').set_weights(dense3_reset.get_weights())
    model.get_layer(name = 'result_output').set_weights(rout_reset.get_weights())
    model.get_layer(name = 'score_output').set_weights(sout_reset.get_weights())
    return(model)


def transfer_model_2(version_path):
    import os
    from logging import getLogger
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    getLogger('tensorflow').setLevel(50)
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import load_model
    model = load_model(version_path)
    inputs = keras.Input(shape=(220,))
    dense3_reset = layers.Dense(100,kernel_initializer= keras.initializers.he_normal)
    rdense4_reset = layers.Dense(30,kernel_initializer=keras.initializers.he_normal)
    rout_reset = layers.Dense(1,kernel_initializer=keras.initializers.he_normal)
    sdense4_reset = layers.Dense(30,kernel_initializer=keras.initializers.he_normal)
    sout_reset = layers.Dense(7,kernel_initializer=keras.initializers.he_normal)
    x = dense3_reset(inputs)
    x1 = rdense4_reset(x)
    x2 = sdense4_reset(x)
    x1 = rout_reset(x1)
    x2 = sout_reset(x2)
    model.get_layer(name = 'rdense_layer3').set_weights(dense3_reset.get_weights())
    model.get_layer(name = 'sdense_layer3').set_weights(dense3_reset.get_weights())
    model.get_layer(name = 'rdense_layer4').set_weights(rdense4_reset.get_weights())
    model.get_layer(name = 'sdense_layer4').set_weights(sdense4_reset.get_weights())
    model.get_layer(name = 'result_output').set_weights(rout_reset.get_weights())
    model.get_layer(name = 'score_output').set_weights(sout_reset.get_weights())
    return(model)

# fit 
def fit_model(model,
              train_x, 
              train_y,
              batch_size,
              epochs,
              validation_percent=.2):
    import os
    from logging import getLogger
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    getLogger('tensorflow').setLevel(50)
    from tensorflow import keras
    from tensorflow.keras import layers

    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = {
            'score_output':keras.losses.CategoricalCrossentropy(),
            'result_output':keras.losses.BinaryCrossentropy()
                },
        metrics = {
            'score_output':[[keras.metrics.CategoricalCrossentropy(),keras.metrics.CategoricalAccuracy()]],
            'result_output':[[keras.metrics.BinaryCrossentropy(),keras.metrics.BinaryAccuracy()]]
                   },
        loss_weights = {
            'score_output':1,
            'result_output':1
                        }
        )
    #fitting
    history = model.fit(train_x,
                        train_y,
                        batch_size = batch_size,
                        validation_split=validation_percent,
                        epochs = epochs,
                        verbose = 1)
    return(history)







if __name__=='__main__':
    pass
    # import tensorflow as tf
    # from tensorflow import keras
    # from tensorflow.keras import layers
    # ex = np.array([np.zeros((7,6,2))])
    # ex1 = np.zeros((7,6,2))
    # testnet = make_model()
    # sconet = make_model_scores_only()
    # testnet = keras.models.load_model('C:/Users/matth/python/connectfour/c4netMT_versions/c4netMT_0-5')
    # testmodel = tf.function(testnet,
    #                         input_signature = [tf.TensorSpec(testnet.inputs[0].shape, 
    #                                                          testnet.inputs[0].dtype)],
    #                         jit_compile=True)
    # testmodel = tf.function(testnet,
    #                         input_signature = [tf.TensorSpec(testnet.inputs[0].shape, 
    #                                                          testnet.inputs[0].dtype)])
    # tmconcrete = testmodel.get_concrete_function()
    # tmconcrete = testmodel.get_concrete_function(tf.TensorSpec(shape=testnet.inputs[0].shape, dtype=testnet.inputs[0].dtype))
    # converter = tf.lite.TFLiteConverter.from_concrete_functions([tmconcrete])
    # tflite_model = converter.convert()
    # testint = tf.lite.Interpreter(model_content=tflite_model)
    # converter = tf.lite.TFLiteConverter.from_saved_model('C:/Users/matth/python/connectfour/c4netMT_versions/c4netMT_0-5')
    # tmlite = lite_model_class.from_concrete_function(tmconcrete)
    # tflite_model = converter.convert()


    # toc2 =tt()
    # testnet = keras.models.load_model('C:/Users/matth/python/connectfour/c4netMT_versions/c4netMT_0-5')
    # testmodel = tf.function(testnet,
    #                         input_signature = [tf.TensorSpec(testnet.inputs[0].shape, 
    #                                                           testnet.inputs[0].dtype)])
    # tmconcrete = testmodel.get_concrete_function()
    # tmlite2 = lite_model_class.from_concrete_function(tmconcrete)
    # tic2=tt()
    # toc1 = tt()
    # testnet = keras.models.load_model('C:/Users/matth/python/connectfour/c4netMT_versions/c4netMT_0-5')
    # tmlite1 = lite_model_class.from_keras_model(testnet)
    # tic1 = tt()
    
    # testmodel1 = tf.function(testnet,
    #                         input_signature = [tf.TensorSpec(testnet.inputs[0].shape, 
    #                                                           testnet.inputs[0].dtype)], 
    #                         jit_compile=True)

    
    
    # toc = tt()
    # for n in range(10000):
    #     tmlite1.predict(ex1)
    # tic = tt()
    # print('Lite From keras model:',(tic-toc)/10000)
    
    # toc = tt()
    # for n in range(10000):
    #     tmlite2.predict(ex1)
    # tic = tt()
    # print('Lite From concrete function:',(tic-toc)/10000)
    
    
    # toc = tt()
    # for n in range(100):
    #     testmodel(ex)
    # tic = tt()
    # print('tf.function no jit:',(tic-toc)/100)

    # toc = tt()
    # for n in range(100):
    #     testmodel1(ex)
    # tic = tt()
    # print('tf.function jit:',(tic-toc)/100)

    # toc = tt()
    # for n in range(100):
    #     testnet(ex)
    # tic = tt()
    # print('Direct call to model:',(tic-toc)/100)
    
    # toc = tt()
    # for n in range(100):
    #     testnet.predict(ex)
    # tic = tt()
    # print('Predict method:',(tic-toc)/100)
    
    # print(tmlite1.predict(ex1))
    # print(tmlite2.predict(ex1))
    # print(testmodel(ex))
    
    # print(tic1-toc1)
    # print(tic2-toc2)

    # @tf.function
    # def evaluate_model(model,data):
    #     tf.convert_to_tensor(data)
    #     return(model(data))
    # testmodel(ex)
