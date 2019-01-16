import numpy as np
import os
import random


from skimage.io import imread

class Data_Generator_from_folder(object):
    def __init__(self, image_path, anno_path, batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, n_class):
        self.image_path = image_path
        self.anno_path = anno_path
        self.batch_size = batch_size
        self.ids = os.listdir(self.image_path)
        random.shuffle(self.ids)
        self.DATA_LENGTH = len(self.ids)
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_DEPTH = IMG_DEPTH
        self.IMG_CHANNELS = IMG_CHANNELS
        self.n_class = n_class
    
    def generator(self):
        while 1:
            for idx in range(0, self.DATA_LENGTH, self.batch_size):
                ids_ = self.ids[idx:idx+self.batch_size]
                img = np.zeros((self.batch_size, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_DEPTH, self.IMG_CHANNELS), dtype=np.uint8)
                mask = np.zeros((self.batch_size, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_DEPTH, self.n_class), dtype=np.uint8)
                i=0
                for id_ in ids_:
                    img_ = imread(self.image_path + id_, dtype=np.uint8)
                    img_ = np.expand_dims(img_, -1)
                    img[i] = img_
                    
                    mask_ = imread(self.anno_path + id_, dtype=np.uint8)
                    exist_class = np.unique(mask_).astype(np.uint8)
                    if exist_class[0] == 0:
                        exist_class = np.delete(exist_class,0)
                    for cls in exist_class:
                        class_mask = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_DEPTH), dtype=np.uint8)
                        class_mask[mask_ == cls] = 1
                        mask[i,:,:,:, cls-1] = class_mask
                    i += 1
                yield img, mask