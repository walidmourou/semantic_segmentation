from keras.models import Model, load_model
from keras.layers import Input, Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.losses import binary_crossentropy

import tensorflow as tf
from tensorflow.python.client import device_lib

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
from tqdm import tqdm

import time
import datetime

import random


class UNet_keras():
    def __init__(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, N_CLASS):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        self.N_CLASS = N_CLASS

    def graph(self, IN_FILTERS = 32, KERNEL_INIT = 'he_normal', VERBOSE = False, weights=''):
        # Build U-Net model
        inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        s = Lambda(lambda x: x / 255.0) (inputs)
        c1 = Conv2D(IN_FILTERS, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (s)
        c1 = Dropout(0.1) (c1)
        c1 = Conv2D(IN_FILTERS, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)
        c2 = Conv2D(IN_FILTERS*2, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (p1)
        c2 = Dropout(0.1) (c2)
        c2 = Conv2D(IN_FILTERS*2, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)
        c3 = Conv2D(IN_FILTERS*4, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (p2)
        c3 = Dropout(0.2) (c3)
        c3 = Conv2D(IN_FILTERS*4, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)
        c4 = Conv2D(IN_FILTERS*8, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (p3)
        c4 = Dropout(0.2) (c4)
        c4 = Conv2D(IN_FILTERS*8, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        c5 = Conv2D(IN_FILTERS*16, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (p4)
        c5 = Dropout(0.3) (c5)
        c5 = Conv2D(IN_FILTERS*16, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (c5)
        u6 = Conv2DTranspose(IN_FILTERS*8, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(IN_FILTERS*8, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (u6)
        c6 = Dropout(0.2) (c6)
        c6 = Conv2D(IN_FILTERS*8, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (c6)
        u7 = Conv2DTranspose(IN_FILTERS*4, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(IN_FILTERS*4, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (u7)
        c7 = Dropout(0.2) (c7)
        c7 = Conv2D(IN_FILTERS*4, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (c7)
        u8 = Conv2DTranspose(IN_FILTERS*2, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(IN_FILTERS*2, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (u8)
        c8 = Dropout(0.1) (c8)
        c8 = Conv2D(IN_FILTERS*2, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (c8)
        u9 = Conv2DTranspose(IN_FILTERS, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(IN_FILTERS, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (u9)
        c9 = Dropout(0.1) (c9)
        c9 = Conv2D(IN_FILTERS, (3, 3), activation='elu', kernel_initializer=KERNEL_INIT, padding='same') (c9)
        outputs = Conv2D(self.N_CLASS, (1, 1), activation='sigmoid') (c9)
        model = Model(inputs=[inputs], outputs=[outputs])
        if VERBOSE:
            model.summary()
        if weights:
            model.load_weights(weights)
        return model

    def focal_loss(self, y_true, y_pred):
        gamma=2
        alpha=0.25
        
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(tf.equal(y_true,1), alpha_factor, 1-alpha_factor)

        focal_weight = tf.where(tf.equal(y_true,1), 1-y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        return focal_weight * binary_crossentropy(y_true, y_pred)

    def dice_coe_loss(self, y_true, y_pred ): # 
        smooth=1e-5
        intersection = K.sum(y_true * y_pred, axis=-1)
        # dice = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
        dice = (2. * intersection + smooth) / (K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) + smooth)
        return 1 - dice
    
    def mean_iou_metric(self, y_true, y_pred):
        y_pred_ = tf.to_int32(y_pred > 0.5)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_,self.N_CLASS)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        return score

    def train(self, model, training_generator, validation_generator, BATCH_SIZE=8, NB_EPOCHS=500, LR=0.001, loss='crossentropy', num_gpu=[0]):
        # Define Callbacks
        TRAIN_NAME = 'unet_' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
        earlystopper = EarlyStopping(patience=10, verbose=1)
        checkpointer = ModelCheckpoint(TRAIN_NAME + '.h5', verbose=1, save_best_only=True)
        tbCallBack = TensorBoard(log_dir='logs/' + TRAIN_NAME, histogram_freq=0, write_graph=True, write_images=True)
                                
        #compile the model with gpu
        if len(num_gpu)>1:
            model = multi_gpu_model(model, gpus=num_gpu)
        else:
            tf.device('/gpu:'+str(num_gpu[0]))

        sgd = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
        # Loss selection
        if loss == 'dice_coe':
            model.compile(optimizer=sgd, loss=self.dice_coe_loss, metrics=[self.mean_iou_metric])
        elif loss == 'crossentropy':
            model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[self.mean_iou_metric])
        elif loss == 'focal':
            model.compile(optimizer=sgd, loss=self.focal_loss, metrics=[self.mean_iou_metric])

        # Train model on dataset
        model.fit_generator(generator=training_generator.generator(),
                            validation_data=validation_generator.generator(),
                            epochs=NB_EPOCHS,
                            steps_per_epoch=training_generator.DATA_LENGTH//BATCH_SIZE,
                            validation_steps=validation_generator.DATA_LENGTH//BATCH_SIZE,
                            callbacks=[earlystopper, checkpointer, tbCallBack])


    def inference(self, model, ckpt_path, image_path, mask_path):
        image = imread(image_path)
        # self.mask = imread(self.path_mask)
        mask = np.fromfile(mask_path, dtype=np.uint8).reshape((1600,1600,1600))
        mask[mask == 7] = 0
        mask[mask == 6] = 0
        list_idx = []
        for k in range(len(image)):
            if len(np.nonzero(mask[k,:,:])[0]):
                list_idx.append(k)
        s_idx = random.choice(list_idx)
        
        image_ = image[s_idx]
        #########################""
        image_ = np.rint(resize(image_, (self.IMG_HEIGHT, self.IMG_WIDTH), preserve_range=True)).astype(np.uint8)
        
        mask_ = mask[s_idx]
        #########################""
        mask_ = np.rint(resize(mask_, (self.IMG_HEIGHT, self.IMG_WIDTH), preserve_range=True)).astype(np.uint8)

        image_ = np.expand_dims(image_, axis=0)
        image_ = np.expand_dims(image_, axis=3)

        # model.load_model(ckpt_path)
        pmask_ = model.predict(image_)
        pmask_[pmask_>0.5] = 1

        fmask = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH))
        for i in range(5):
            fmask += pmask_[0,:,:,i]*(i+1)

        print('max value in mask', fmask.max())
        fig = plt.figure(figsize=(9, 3))

        fig.add_subplot(1, 3, 1)
        plt.imshow(image_[0,:,:,0], cmap='gray')
        plt.title("Original Image")

        fig.add_subplot(1, 3, 2)
        plt.imshow(fmask)
        plt.title("Infered mask")

        fig.add_subplot(1, 3, 3)
        plt.imshow(mask_)
        plt.title("True mask")
        plt.show()