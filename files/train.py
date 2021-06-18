'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Training script for MNIST multi-headed classifier.
'''

'''
Author: Mark Harvey
'''

import os
import shutil
import sys
import argparse
import numpy as np


# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.datasets import mnist

from custom_cnn_mh import customcnn
from utils import mnist_download


DIVIDER = '-----------------------------------------'


def train(input_height,input_width,input_chan,batchsize,learnrate,epochs,chkpt_dir,tboard):

 
    '''
    Prepare MNIST dataset
    '''
    print('Preparing dataset..')
    x_train,  x_test,  y_train, y_test,\
    y_train0, y_test0, y_train1, y_test1, \
    y_train2, y_test2, y_train3, y_test3, \
    y_train4, y_test4, y_train5, y_test5, \
    y_train6, y_test6, y_train7, y_test7, \
    y_train8, y_test8, y_train9, y_test9 = mnist_download()



    '''
    Define the model
    '''
    print('Defining the model..')
    classes = np.size(np.unique(y_train))
    print('Found',classes,'classes..')
    model = customcnn(input_shape=(input_height,input_width,input_chan),classes=classes,filters=[16,32,64,128])

    print('\n'+DIVIDER)
    print(' Model Summary')
    print(DIVIDER)
    print(model.summary())
    print("Model Inputs: {ips}".format(ips=(model.inputs)))
    print("Model Outputs: {ops}".format(ops=(model.outputs)))
    
    
    '''
    Call backs
    '''
    print('Setting up callbacks..')
    tb_call = TensorBoard(log_dir=tboard)

    chkpt_call = ModelCheckpoint(filepath=os.path.join(chkpt_dir,'f_model.h5'), 
                                 verbose=1,
                                 save_best_only=True)

    callbacks_list = [tb_call, chkpt_call]


    '''
    Compile model
    Adam optimizer to change weights & biases
    Loss function is binary crossentropy
    '''
    # build dictionaries of loss functions and loss weights
    losses = {}
    loss_weights = {}
    for i in range(classes):
      losses['out_'+str(i)] = BinaryCrossentropy(from_logits=True)
      loss_weights['out_'+str(i)] = 1.0

    model.compile(optimizer=Adam(learning_rate=learnrate),
                  loss=losses, loss_weights=loss_weights,
                  metrics=['accuracy'])
 
    '''
    Training
    '''
    print('\n'+DIVIDER)
    print(' Training model with training set..')
    print(DIVIDER)

    # make folder for saving trained model checkpoint
    os.makedirs(chkpt_dir, exist_ok = True)

    # run training
    train_dict = {'out_0':y_train0, 'out_1':y_train1, 'out_2':y_train2, 'out_3':y_train3,
                  'out_4':y_train4, 'out_5':y_train5, 'out_6':y_train6, 'out_7':y_train7,
                  'out_8':y_train8, 'out_9':y_train9 }

    test_dict = {'out_0':y_test0, 'out_1':y_test1, 'out_2':y_test2, 'out_3':y_test3,
                 'out_4':y_test4, 'out_5':y_test5, 'out_6':y_test6, 'out_7':y_test7,
                 'out_8':y_test8, 'out_9':y_test9 }


    train_history=model.fit(x = x_train,
                            y = train_dict,
                            batch_size=batchsize,
                            epochs=epochs,
                            validation_data=(x_test,test_dict),
                            callbacks=callbacks_list,
                            verbose=0)

    '''
    Evaluate best checkpoint
    '''
    print('\n'+DIVIDER)
    print(' Load and evaluate best checkpoint..')
    print(DIVIDER)

    model = load_model(os.path.join(chkpt_dir,'f_model.h5'), compile=True)
    model.compile(optimizer=Adam(learning_rate=learnrate),
                  loss=losses, loss_weights=loss_weights,
                  metrics=['accuracy'])
    scores = model.evaluate(x=x_test, y=test_dict,
                            batch_size=batchsize,
                            verbose=0
                            )
    for i in range(classes):
      print(' Classifier',str(i),'accuracy: {0:.4f}'.format(scores[i+classes+1]*100),'%')

    print("\nTensorBoard can be opened with the command: tensorboard --logdir={dir} --host localhost --port 6006".format(dir=tboard))


    '''
    Run some predictions
    '''
    print('\n'+DIVIDER)
    print(' Predictions..')
    print(DIVIDER)

    predictions = model.predict(x=x_test,
                                batch_size=1,
                                verbose=0 )
    
    results = []
    for i in range(x_test.shape[0]):
      test = np.zeros(classes)
      for j in range(classes):
        test[j] = predictions[j][i]
      results.append(np.argmax(test))


    results = np.array(results)
    matches = np.count_nonzero(results==y_test)
    acc = matches/x_test.shape[0]

    print(' Prediction accuracy: {0:.1f}'.format(acc*100),'%')

    print('*** FINISHED ***')
  
    
    return




def run_main():
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-ih','--input_height',type=int,   default=28,   help='Input image height in pixels.')
    ap.add_argument('-iw','--input_width', type=int,   default=28,   help='Input image width in pixels.')
    ap.add_argument('-ic','--input_chan',  type=int,   default=1,    help='Number of input image channels.')
    ap.add_argument('-b', '--batchsize',   type=int,   default=100,  help='Training batchsize. Must be an integer. Default is 100.')
    ap.add_argument('-e', '--epochs',      type=int,   default=65,   help='number of training epochs. Must be an integer. Default is 65.')
    ap.add_argument('-lr','--learnrate',   type=float, default=0.0001,  help='optimizer learning rate. Must be floating-point value. Default is 0.0001')
    ap.add_argument('-cf','--chkpt_dir',   type=str,   default='float_model', help='Path and name of folder for storing Keras checkpoints. Default is float_model')
    ap.add_argument('-tb','--tboard',      type=str,   default='tb_logs', help='path to folder for saving TensorBoard data. Default is tb_logs.')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('Keras version      : ',tf.keras.__version__)
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--input_height : ',args.input_height)
    print ('--input_width  : ',args.input_width)
    print ('--input_chan   : ',args.input_chan)
    print ('--batchsize    : ',args.batchsize)
    print ('--learnrate    : ',args.learnrate)
    print ('--epochs       : ',args.epochs)
    print ('--chkpt_dir    : ',args.chkpt_dir)
    print ('--tboard       : ',args.tboard)
    print(DIVIDER)
    

    train(args.input_height,args.input_width,args.input_chan, \
          args.batchsize,args.learnrate,args.epochs,args.chkpt_dir,args.tboard)


if __name__ == '__main__':
    run_main()
