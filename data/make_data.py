#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on July 18, 2021, 23:57:23
@author: dianwen
"""
import pandas as pd
import json
import sys
import os

def make_manifest(data_path, csv_path='data', phase='train'):
    """ Make training and testing JSON manifest
    Input args: 
        datapath: path dir to chest CXR, type=str
        csv_path: path dir to labels, type=str
        phase:'train' or test, type=str
    """
    ## load dataset
    data_csv = pd.read_csv(os.path.join(csv_path, 'Data_Entry_2017.csv'))
    radiologist_labelled = pd.read_csv(os.path.join(csv_path, 'NIH_radiologist_labelled_positive.csv'))
    radiologist_labelled = radiologist_labelled.rename(columns={'filename': 'Image Index'})
    
    ## load official train/test split 
    train_set = pd.read_csv(os.path.join(csv_path, 'train_test_splits/train_val_list.txt'), header=None, names=['Image Index'])
    train_set = train_set.join(data_csv.set_index('Image Index'), on='Image Index', how='inner')
    
    test_set = pd.read_csv(os.path.join(csv_path, 'train_test_splits/test_list.txt'), header=None, names=['Image Index'])
    test_set = test_set.join(data_csv.set_index('Image Index'), on='Image Index', how='inner')
    
    ## make training manifest
    if phase == 'train':
        negative = train_set[['pneumothorax' not in file for file in train_set['Finding Labels'].str.lower()]]
        negative = list(set(negative['Image Index']) - set(radiologist_labelled['Image Index']))

        positive = train_set.join(radiologist_labelled.set_index('Image Index'), on='Image Index', how='inner')
        positive = list(set(positive['Image Index']))
        
        ## train-valid split
        negative = negative[:int(len(negative) * 0.95)]
        positive = positive[:int(len(positive) * 0.95)]
            
    elif phase == 'valid':
        negative = train_set[['pneumothorax' not in file for file in train_set['Finding Labels'].str.lower()]]
        negative = list(set(negative['Image Index']) - set(radiologist_labelled['Image Index']))

        positive = train_set.join(radiologist_labelled.set_index('Image Index'), on='Image Index', how='inner')
        positive = list(set(positive['Image Index']))
        
        ## train-valid split
        negative = negative[int(len(negative) * 0.95):]
        positive = positive[int(len(positive) * 0.95):]
        
    elif phase == 'test':
        negative = test_set[['pneumothorax' not in file for file in test_set['Finding Labels'].str.lower()]]
        negative = list(set(negative['Image Index']) - set(radiologist_labelled['Image Index']))

        positive = test_set.join(radiologist_labelled.set_index('Image Index'), on='Image Index', how='inner')
        positive = list(set(positive['Image Index']))
        
    else:
        print("Error in arg phase. Function only considers 'train', 'valid' or 'test'.")
        
    ## make json dataset
    with open(os.path.join(csv_path, str(phase) + '_manifest.json'), 'w') as jout:
        for fname in positive:
            # Write the metadata to the manifest
            metadata = {
                "image_filepath": os.path.join(data_path, fname),
                "label": "positive"}

            json.dump(metadata, jout)
            jout.write('\n')
            jout.flush()
        
        for fname in negative:
            # Write the metadata to the manifest
            metadata = {
                "image_filepath": os.path.join(data_path, fname),
                "label": "negative"}

            json.dump(metadata, jout)
            jout.write('\n')
            jout.flush()
    
if __name__ == "__main__":
    ## assert requirement for path to dataset
    if len(sys.argv) < 2:
        print('Error in creating data manifest. Please check that you provide NIH CXR root dir.')
    
    if len(sys.argv) <= 2:
        csv_path = sys.argv[2]
    else: ## set default csv path
        csv_path = 'data'

    data_path = sys.argv[1]
    
    make_manifest(data_path=data_path, csv_path=csv_path, phase='train')
    make_manifest(data_path=data_path, csv_path=csv_path, phase='valid')
    make_manifest(data_path=data_path, csv_path=csv_path, phase='test')