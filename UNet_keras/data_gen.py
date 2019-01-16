import numpy as np
import os
import random


from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize, rotate
from skimage.filters import gaussian
from skimage.morphology import label


class Data_Generator_from_folder(object):
    def __init__(self, image_path, anno_path, batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, n_class):
        self.image_path = image_path
        self.anno_path = anno_path
        self.batch_size = batch_size
        self.ids = os.listdir(self.image_path)
        random.shuffle(self.ids)
        self.DATA_LENGTH = len(self.ids)
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        self.n_class = n_class
    
    def crop_or_pad(self, IMG, H, W):
        sze = IMG.shape
        h_dif_l = abs(H-sze[0])//2 + abs(H-sze[0])%2
        h_dif_r = abs(H-sze[0])//2

        w_dif_l = abs(W-sze[1])//2 + abs(W-sze[1])%2
        w_dif_r = abs(W-sze[1])//2
        
        if (sze[0]<H and sze[1]<W):
            if len(sze)==3:
                IMG = np.pad(IMG, ((h_dif_l, h_dif_r), (w_dif_l, w_dif_r), (0, 0)), mode='constant', constant_values=0)
            else:
                IMG = np.pad(IMG, ((h_dif_l, h_dif_r), (w_dif_l, w_dif_r)), mode='constant', constant_values=0)
        elif (sze[0]<H or sze[1]<W):
            if (sze[0]<H):
                if len(sze)==3:
                    IMG = np.pad(IMG, ((h_dif_l, h_dif_r), (0, 0), (0, 0)), mode='constant', constant_values=0)
                    IMG = IMG[:, w_dif_l:sze[1]-w_dif_r, :]
                else:
                    IMG = np.pad(IMG, ((h_dif_l, h_dif_r), (0, 0)), mode='constant', constant_values=0)
                    IMG = IMG[:, w_dif_l:sze[1]-w_dif_r]
            else:
                if len(sze)==3:
                    IMG = np.pad(IMG, ((0, 0), (w_dif_l, w_dif_r), (0, 0)), mode='constant', constant_values=0)
                    IMG = IMG[h_dif_l:sze[0]-h_dif_r, :, :]
                else:
                    IMG = np.pad(IMG, ((0, 0), (w_dif_l, w_dif_r)), mode='constant', constant_values=0)
                    IMG = IMG[h_dif_l:sze[0]-h_dif_r, :]
        else:
            if len(sze)==3:
                IMG = IMG[h_dif_l:sze[0]-h_dif_r, w_dif_l:sze[1]-w_dif_r, :]
            else:
                IMG = IMG[h_dif_l:sze[0]-h_dif_r, w_dif_l:sze[1]-w_dif_r]
        return IMG
    
    def generator(self):
        while 1:
            for idx in range(0, self.DATA_LENGTH, self.batch_size):
                ids_ = self.ids[idx:idx+self.batch_size]
                img = np.zeros((self.batch_size, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
                mask = np.zeros((self.batch_size, self.IMG_HEIGHT, self.IMG_WIDTH, self.n_class), dtype=np.uint8)
                i=0
                for id_ in ids_:
                    img_ = imread(self.image_path + id_, dtype=np.uint8)
                    if img_.shape[0] == 3 :
                        img_ = np.rollaxis(img_, 3, 1)
                    if (len(img_.shape) == 2 and self.IMG_CHANNELS == 3):
                        img_ = img_[:,:,None] * np.ones(3)[None, None,:]
                    elif (len(img_.shape) == 2 and self.IMG_CHANNELS == 1):
                        img_ = np.expand_dims(img_, axis=2)
                    
                    #########################""
                    img_ = np.rint(resize(img_, (self.IMG_HEIGHT, self.IMG_WIDTH), preserve_range=True)).astype(np.uint8)
                    # img_ = self.crop_or_pad(img_, self.IMG_HEIGHT, self.IMG_WIDTH)
                    
                    img[i] = img_
                    
                    mask_ = imread(self.anno_path + id_.split('.')[0] + '.png', dtype=np.uint8)
                    
                    #########################""
                    mask_ = np.rint(resize(mask_, (self.IMG_HEIGHT, self.IMG_WIDTH), preserve_range=True)).astype(np.uint8)
                    # mask_ = self.crop_or_pad(mask_, self.IMG_HEIGHT, self.IMG_WIDTH)
                    
                    exist_class = np.unique(mask_).astype(np.uint8)
                    if exist_class[0] == 0:
                        exist_class = np.delete(exist_class,0)
                    for cls in exist_class:
                        class_mask = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH), dtype=np.uint8)
                        class_mask[mask_ == cls] = 1
                        mask[i,:,:, cls-1] = class_mask
                    i += 1
                yield img, mask


class Data_Generator_from_cube(object):
    def __init__(self, path_image, path_mask, batch_size, IMG_HEIGHT, IMG_WIDTH, n_channel, n_class, convert_1class=False):
        self.path_image = path_image
        self.path_mask = path_mask
        self.batch_size = batch_size
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.n_channel = n_channel
        self.n_class = n_class
        self.convert_1class = convert_1class
        self.DATA_LENGTH = 1600

    def transformer(self, s_idx):
        angl = random.choice([0, 90, 180, 270])
        blr = random.choice([0, 1, 2, 3])
        flp = random.choice(['n', 'v', 'h'])
        
        img = self.image[s_idx] 
        mask = self.mask[s_idx]

        #########################""
        img = np.rint(resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), preserve_range=True)).astype(np.uint8)
        #########################""
        mask = np.rint(resize(mask, (self.IMG_HEIGHT, self.IMG_WIDTH), preserve_range=True)).astype(np.uint8)
        
        img = gaussian(img, blr)
        # mask = gaussian(mask, blr)

        img = rotate(img, angl)
        mask = rotate(mask, angl)

        if flp == 'v':
            img = np.flipud(img)
            mask = np.flipud(mask)
        elif flp == 'h':
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        
        img = np.rint(img*255)
        img[img>255] = 255
        img[img<0] = 0
        img = img.astype(np.uint8)
        
        mask = np.rint(mask*255)
        mask[mask>255] = 255
        mask[mask<0] = 0
        mask = mask.astype(np.uint8) 
        
        return img, mask

    def generator(self):
        self.image = imread(self.path_image)
        # self.mask = imread(self.path_mask)
        self.mask = np.fromfile(self.path_mask, dtype=np.uint8).reshape((1600,1600,1600))
        self.mask[self.mask == 7] = 0
        self.mask[self.mask == 6] = 0
        if self.convert_1class:
            self.mask[self.mask > 0] = 1
            self.n_class = 1

        # self.IMG_HEIGHT, self.IMG_WIDTH = self.image[0,:,:].shape
        
        list_idx = []
        for k in range(len(self.image)):
            if len(np.nonzero(self.mask[k,:,:])[0]):
                list_idx.append(k)
        while 1:
            img = np.zeros((self.batch_size, self.IMG_HEIGHT, self.IMG_WIDTH, self.n_channel), dtype=np.uint8)
            msk = np.zeros((self.batch_size, self.IMG_HEIGHT, self.IMG_WIDTH, self.n_class), dtype=np.uint8)
            for idx in range(self.batch_size):
                s_idx = random.choice(list_idx)
                img_, mask_ = self.transformer(s_idx)
                img[idx,:,:,0] = img_ # We should replace 0 by the appropriate param (generalize to color) 
                
                for cls in range(0, self.n_class):
                    class_mask = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH), dtype=np.uint8)
                    class_mask[mask_==cls+1] = 1
                    msk[idx,:,:, cls] = class_mask
          
            yield img, msk