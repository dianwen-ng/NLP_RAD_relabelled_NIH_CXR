#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Feb 4, 2020, 23:44:06
@author: dianwen
"""

import random
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import tensorflow as tf
import keras
from albumentations import (
    Compose, HorizontalFlip, CLAHE, OneOf, Resize, Normalize,
    RandomBrightness, RandomContrast, RandomGamma, ToFloat,
    ShiftScaleRotate,GridDistortion, ElasticTransform,
    RandomBrightness, RandomContrast, OpticalDistortion)

## ImageNet Statistics: Data mean and standard deviation
imagenet_stats = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225)
    }

def get_transforms(phase, 
                   mean=imagenet_stats['mean'],
                   std=imagenet_stats['std']):
    """ Augmentation helpers
    Input args: 
        phase: 'train' or 'valid', type:str
        mean: list of float
        std: list of float
    Function helps to perform augmentation steps for training model.
    """
    assert isinstance(phase, str), 'Argument of wrong type. Input takes only "train" or "valid".'
    if phase == 'train':
        ## augmentation for training batch
        augmentation_list = Compose([
            ## random horizontal flip
            HorizontalFlip(p=0.5),
            
            ## random brightness
            RandomBrightness(p=0.35),
            
            ## random adjust contrast
            OneOf([
                RandomContrast(limit=0.5),
                RandomGamma(),
                CLAHE(clip_limit=2.0)
            ], p=0.3),
            
            ## random image distortion/warping
            OneOf([
                ElasticTransform(alpha=120, sigma=100*0.16, alpha_affine=100*0.04),
                GridDistortion(),
                OpticalDistortion(distort_limit=1.0, shift_limit=0.05),
            ], p=0.25),
            
            ## random zoom and rotate 
            ShiftScaleRotate(shift_limit=0,
                             scale_limit=0.1,
                             rotate_limit=10,
                             p=0.5,
                             border_mode=cv2.BORDER_CONSTANT),
            
            ## make image normalization
            Normalize(mean=mean,std=std, p=1),
            
            ## convert to float
            ToFloat()])
        
    elif phase == 'valid' or phase == 'test':
        ## augmentation for testing/validation batch
        augmentation_list = Compose([
            Normalize(mean=mean, std=std, p=1),
            ToFloat()], p=1)
        
    else:
        return 'No such augmenting option. i.e. phase can only be "train", "valid" or "test"' 

    return augmentation_list


class NIHDataGenerator(keras.utils.Sequence):
    def __init__(self, json_file, augmentations, 
                 image_size=512, balance_ratio=0.5, shuffle=True,
                 batch_size=32):
        
        self.data = np.array(open(json_file).readlines())
        self.transforms = augmentations
        self.image_size = image_size
        self.balance_ratio = balance_ratio
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        if balance_ratio != None:
            ## get number of positive and negative samples per iteration
            self.num_positive_per_step = int(batch_size * balance_ratio)
            self.num_negative_per_step = int(batch_size - self.num_positive_per_step)
        
        self.labels = np.array([int(eval(item)['label'] == 'positive') for item in self.data])
        self.positive_indices = np.arange(len(self.data))[self.labels == 1]
        self.negative_indices = np.arange(len(self.data))[self.labels == 0]
        
        self.nsteps = len(self.negative_indices) // self.num_negative_per_step
        self.on_epoch_end()
        
    def __getitem__(self, index):
        ## get indices for each iteration of batch samples.
        pos_start_index = int(index % (len(self.positive_indices) // self.num_positive_per_step))
        pos_sample_indices = self.pos_idx[pos_start_index * self.num_positive_per_step:\
                                          min((pos_start_index + 1) * self.num_positive_per_step, len(self.positive_indices))]
        
        neg_start_index = int(index % (len(self.negative_indices) // self.num_negative_per_step))
        neg_sample_indices = self.neg_idx[neg_start_index * self.num_negative_per_step:\
                                          min((neg_start_index + 1) * self.num_negative_per_step, len(self.negative_indices))]
        
        ## find list of samples 
        batch_im_path = self.data[neg_sample_indices].tolist() + self.data[pos_sample_indices].tolist()
        batch_label = self.labels[neg_sample_indices].tolist() + self.labels[pos_sample_indices].tolist()
        
        ## read images
        X = self.image_generation(batch_im_path)
        
        im = [] ## augmentation steps here
        for img in X:
            augmented = self.transforms(image=img)
            im.append(augmented['image'])

        return np.array(im), np.array(batch_label, dtype='float32')
    
    def __len__(self):
        if self.balance_ratio != None: 
            return int(self.nsteps) - 1
        else:
            return len(self.data) // self.batch_size
    
    def on_epoch_end(self):
        ## shuffle dataset for next epoch
        if self.balance_ratio != None:
            self.pos_idx = self.positive_indices.copy()
            self.neg_idx = self.negative_indices.copy()
            
            if self.shuffle:
                np.random.shuffle(self.pos_idx)
                np.random.shuffle(self.neg_idx)
        else:
            self.label_index = np.arange(len(self.data))
            if self.shuffle:
                np.random.shuffle(self.label_index)
                   
    def image_generation(self, image_paths):
        ## initialization of input image placeholder per step
        X = np.empty((self.batch_size, self.image_size, self.image_size, 3))
        for i, impath in enumerate(image_paths):
            
            im = cv2.imread(eval(impath)['image_filepath'])
            
            ## resize sample
            X[i,] = cv2.resize(im, (self.image_size, self.image_size))
            
        return np.uint8(X)
    
    
if __name__ == "__main__":
    
    np.random.seed(123)
    batch_size=32
    
    generator = NIHDataGenerator("data/train_manifest.json", get_transforms('train'),
                                 image_size=768, balance_ratio=0.5, 
                                 shuffle=True, batch_size=batch_size)
    

    
