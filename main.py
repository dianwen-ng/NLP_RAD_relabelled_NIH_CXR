#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from time import time
import random
import numpy as np
import argparse
import dataloader
import models
from sklearn.utils import class_weight
from keras.optimizers import Adam
from keras.metrics import binary_accuracy
from keras.losses import binary_crossentropy
import tensorflow as tf
import keras
from keras import backend as K

# callbacks params and configs
from keras.callbacks import (ReduceLROnPlateau, LearningRateScheduler,
                             CSVLogger, EarlyStopping, ModelCheckpoint)

## argument parser
def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    ## data dir to JSON file
    parser.add_argument("-data_train", type=str, default='data/train_manifest.json', help="training data json")
    parser.add_argument("-data_val", type=str, default='data/valid_manifest.json', help="validation data json")
    parser.add_argument("-data_test", type=str, default='data/test_manifest.json', help="testing data json")
    
    ## model dataset 
    parser.add_argument('-train_batch_size', action="store_true", default=12)
    parser.add_argument('-valid_batch_size', action="store_true", default=12)
    parser.add_argument('-image_size', action="store_true", default=768)
    parser.add_argument('-bal_ratio', action="store_true", default=0.2)
    
    parser.add_argument('-num_classes', action="store_true", default=1) ## positive vs non-positive
    parser.add_argument('-num_gpu', action="store_true", default=3)
    
    parser.add_argument('-save_dir', type=str, default='save_models')
    parser.add_argument('-num_epochs', action="store_true", default=15)

    parser.add_argument('-lr', action="store_true", default=1e-5)
    parser.add_argument('-model', type=str, required=True)
    
    args = parser.parse_args()
    return args

## define utils for model training
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred,
                         curve='PR',
                         summation_method='careful_interpolation')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, path, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.path = path

    def get_callbacks(self, model_prefix='Model'):
        
        ## save checkpoint
        filepath = os.path.join(self.path, 'improved_weights_{epoch:02d}_{val_binary_accuracy:.2f}.hdf5')
        ## saving based on best acc
        checkpointer = ModelCheckpoint(filepath = filepath, verbose=1, save_best_only=True,
                                       save_weights_only=True, monitor='val_binary_accuracy', mode='max')
        ## saving based on lower loss
        checkpointer2 = ModelCheckpoint(filepath = filepath, verbose=1, save_best_only=True,
                                       save_weights_only=True, monitor = 'val_loss', mode='min')

        ## early stoppage based on lower loss
        stop_train = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, verbose=1, mode = 'min')
        
        ## reduce learning rate scheduling
        #schedule_lr = LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        schedule_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        mode = 'min', patience=2, min_lr=1e-9, cooldown=1)
        
        ## csv logger
        logger = CSVLogger(os.path.join(self.path, 'training_log.csv'))

        ## design callbacks
        callback_list = [schedule_lr, stop_train, checkpointer, checkpointer2, logger]
        
        return callback_list

    def _cosine_anneal_schedule(self, t):
        ## t - 1 is used when t has 1-based indexing
        cos_inner = np.pi * (t % (self.T // self.M)) 
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        lr= float(self.alpha_zero / 2 * cos_out)
        
        ## reduce learning rate as training progress 
        if t>5 and not t%5:
            self.alpha_zero *= 0.5
            
        return lr
    
    
if __name__ == "__main__":
   
    args = parse_args()
    
    ## assert existing saving directory
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    
    ## make generator
    train_generator = dataloader.NIHDataGenerator(args.data_train, dataloader.get_transforms('train'),
                                                  image_size=args.image_size, balance_ratio=args.bal_ratio, 
                                                  shuffle=True, batch_size=args.train_batch_size)
    
    valid_generator = dataloader.NIHDataGenerator(args.data_val, dataloader.get_transforms('valid'),
                                                  image_size=args.image_size, balance_ratio=args.bal_ratio, 
                                                  shuffle=True, batch_size=args.valid_batch_size)
    
    ## make model
    if args.model == 'efficientnet':
        model = models.EfficientNetModel(input_shape=(args.image_size, args.image_size, 3),
                                         num_classes=args.num_classes)
        model = models.ModelMGPU(model, args.num_gpu)
        
    elif args.model == 'densenet':
        model = models.DenseNetModel(input_shape=(args.image_size, args.image_size, 3),
                                     num_classes=args.num_classes)
        model = models.ModelMGPU(model, args.num_gpu)
        
    elif args.model == 'resnet':
        model = models.ResNetModel(input_shape=(args.image_size, args.image_size, 3),
                                   num_classes=args.num_classes)
        model = models.ModelMGPU(model, args.num_gpu)
    else:
        print("Error in model selection. System only supports 'efficientnet', 'densenet', 'resnet'.")

    ## set metrics    
    metrics = [binary_accuracy, auc]
    
    ## compile model
    model.compile(optimizer=Adam(lr=args.lr),
                  loss=binary_crossentropy,
                  metrics=metrics) 
    
    ## define callback
    callbacks = SnapshotCallbackBuilder(nb_epochs=args.num_epochs, 
                                        nb_snapshots=1,
                                        init_lr=args.lr,
                                        path=args.save_dir)
    
    ## make weights for weighted loss 
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_generator.labels),
                                                      train_generator.labels)
    
    ## run model training
    print("================ train configs ==================")
    print("Using input image size: ", str(args.image_size))
    print("Using train batch size: ", str(args.train_batch_size))
    print("Using sampling ratio: ", str(args.bal_ratio))
    print("Using model: ", args.model)
    print("Using learning rate: ", str(args.lr))
    print("Train num of epochs: ", str(args.num_epochs))

    print("\n=============== begin training ==================")
    steps = len(train_generator)
    start = time()
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps, 
                        epochs = args.num_epochs, verbose=1, 
                        callbacks = callbacks.get_callbacks(),
                        validation_data=valid_generator,
                        validation_steps=len(valid_generator),
                        class_weight=class_weights,
                        max_queue_size=12,
                        workers=12, use_multiprocessing=False)
    
    print("Train completes with usage time {}s".format(time() - start))
    print("Saving model weights to ", args.save_dir)
    model.save_weights(os.path.join(args.save_dir,'final_weights.hdf5'))
    
    print("\n=============== begin testing ==================")
    from tqdm import tqdm
    test_set = open(args.data_test).readlines()
    image_processor = dataloader.get_transforms('test')
    test_pred = []
    target = []
    for i, meta in enumerate(tqdm(test_set)):
        img_raw = cv2.resize(cv2.imread(eval(meta)['image_filepath']), (768,768))
        trans = image_processor(image=img_raw)
        img = trans_img['image']
        pred = model.predict(np.expand_dims(img, axis=0))
        test_pred.append(pred.tolist()[0])
        target.append(int(eval(meta)['label'] == 'positive'))
        
    ## saving to csv
    pd.DataFrame.from_dict({'model_pred': test_pred,
                            'target': target}).to_csv(os.path.join(args.save_dir, 'testing.csv'), index=False)
        
        