'''
 Copyright 2021 Xilinx Inc.
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
Author: Mark Harvey
'''

import os

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Flatten, Conv2D, Dropout, Add
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Reshape



def cbr(input,filters,kernel_size,strides):
    '''
    Convolution - BatchNorm - ReLU - Dropout
    '''
    net = Conv2D(filters=filters,kernel_size=kernel_size,\
                 kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01), \
                 strides=strides, padding='same')(input)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dropout(rate=0.1)(net)
    return net


def classifier(input,classes,idx):
    '''
    Classifier head
    '''

    # create a conv layer that will reduce feature maps to (1,1,classes)
    h = K.int_shape(input)[1]
    w = K.int_shape(input)[2]
    assert h <= 16, 'Kernel height must be l6 or less'
    assert w <= 16, 'Kernel width must be l6 or less'

    net = Conv2D(filters=classes,kernel_size=(h,w),kernel_initializer='he_uniform', \
                 kernel_regularizer=regularizers.l2(0.01),strides=w, padding='valid',name='cnv_'+str(idx))(input)
    net = Flatten(name="out_"+str(idx))(net)
    return net


def customcnn(input_shape=(None, None, None),classes=None, filters=[8,16,32,64,128]):
    '''
    Arguments:
      input_shape: tuple of integers indicating height, width, channels
      classes    : integer to set number of classes
      filters    : list of filter sizes
    '''

    input_layer = Input(shape=input_shape)
    net = input_layer

    for f in filters:
      net = cbr(net,f,3,1)
      net = cbr(net,f,3,2)

    out_0 = classifier(net,1,0)
    out_1 = classifier(net,1,1)
    out_2 = classifier(net,1,2)
    out_3 = classifier(net,1,3)
    out_4 = classifier(net,1,4)
    out_5 = classifier(net,1,5)
    out_6 = classifier(net,1,6)
    out_7 = classifier(net,1,7)
    out_8 = classifier(net,1,8)
    out_9 = classifier(net,1,9)
     
    return Model(inputs=input_layer, outputs=[out_0,out_1,out_2,out_3,out_4,out_5,out_6,out_7,out_8,out_9])

  