#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Feb 4, 2020, 00:19:06
@author: dianwen
"""
from efficientnet import EfficientNetB3
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D
from keras.layers import BatchNormalization, Dropout, Add
from keras.layers import LeakyReLU, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.densenet import DenseNet201, DenseNet169, DenseNet121

##################################
###     EfficientNet model     ###
##################################

def EfficientNetModel(input_shape, num_classes, 
                  weight='imagenet'):
    
    img_input = Input(shape=input_shape)
    backbone = EfficientNetB3(weights=weight,
                              include_top=False,
                              input_tensor=img_input,
                              input_shape=input_shape)
    
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='sigmoid', name='predictions')(x)
    
    model = Model(inputs=img_input, outputs=predictions)
    if weight is not 'imagenet' and not None:
        print('loading model weights from pretrained...')
        model.load_weights(weight)
    return model



#############################
###    DenseNet model     ###
#############################

def DenseNetModel(input_shape, num_classes, 
                  weight='imagenet',
                  model_type='Dense121'):
    
    img_input = Input(shape=input_shape)
    if model_type == 'Dense121':
        backbone = DenseNet121(weights=weight,
                               include_top=False,
                               input_tensor=img_input,
                               input_shape=input_shape)
    elif model_type == 'Dense169':
        backbone = DenseNet169(weights=weight,
                               include_top=False,
                               input_tensor=img_input,
                               input_shape=input_shape)
    elif model_type == 'Dense201':
        backbone = DenseNet201(weights=weight,
                               include_top=False,
                               input_tensor=img_input,
                               input_shape=input_shape)
    else:
        return 'Error: no such model. Try specifying "Dense121", "Dense169" or "Dense201".'
    
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='sigmoid', name='predictions')(x)
    
    model = Model(inputs=img_input, outputs=predictions)
    if weight is not 'imagenet' and not None:
        print('loading model weights from pretrained...')
        model.load_weights(weight)
    return model


##############################
###      ResNet model      ###
##############################

from keras.applications.resnet50 import ResNet50
def ResNetModel(input_shape, num_classes,
                weight='imagenet'):
    img_input = Input(shape=input_shape)
    backbone = ResNet50(weights=weight,
                        include_top = False,
                        input_tensor=img_input,
                        input_shape=input_shape)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='sigmoid', name='predictions')(x)
    
    model = Model(inputs=img_input, outputs=predictions)
    if weight is not 'imagenet' and not None:
        print('loading model weights from pretrained...')
        model.load_weights(weight)
    return model



# ~~~~~~~~~~ Enable multi GPU training ~~~~~~~~~~~~~~~
from keras.utils import multi_gpu_model
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
           serial-model holds references to the weights in the multi-gpu model.
           '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)
        else:
            #return Model.__getattribute__(self, attrname)
            return super(ModelMGPU, self).__getattribute__(attrname)