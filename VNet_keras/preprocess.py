import numpy as np

import matplotlib.pyplot as plt
from skimage.io import imread, imsave

# # path_mask = '/media/instadeep/DATA/projects/cube_13B/updated_z665_820_y210_560_x250_660_BU645_13B_A_2um_GT.raw'
# # img = np.fromfile(path_mask, dtype=np.uint8).reshape((155,350,410))


img_path = '/media/instadeep/DATA/projects/cube_13B/mini_cube/updated_z665_820_y210_560_x250_660_BU645_13B_A_2um_GT.tif'
mask_path = '/media/instadeep/DATA/projects/cube_13B/mini_cube/mask_z665_820_y210_560_x250_660_BU645_13B_A_2um_GT.raw'


img = imread(img_path)
mask = np.fromfile(mask_path, dtype=np.uint8).reshape((155,350,410))



# #shape 155,350,410


# img = np.pad(img, ((131, 130), (33, 33), (3, 3)), mode='constant', constant_values=0)
# mask = np.pad(mask, ((131, 130), (33, 33), (3, 3)), mode='constant', constant_values=0)

# # plt.imshow(img[63,:,:])
# # plt.show()

# imsave('/media/instadeep/DATA/projects/cube_13B/mini_cube/image_reshaped.tif', img)
# imsave('/media/instadeep/DATA/projects/cube_13B/mini_cube/mask_reshaped.tif', mask)


# img = imread('/media/instadeep/DATA/projects/cube_13B/mini_cube/image_reshaped.tif')
# mask = imread('/media/instadeep/DATA/projects/cube_13B/mini_cube/mask_reshaped.tif')

# La nouvelle classification est la suivante :
# - classe 1 : Planispiralé
# - classe 2 : Bisérié : Textulariidae
# - classe 3 : Trisérié ou multisérié 
# - classe 4 : test a structure cloisonnaire
# - classe 5 : Miliolidae 
# - classe 6 : non fossile 
# - classe 7 : porosité
# - classe 8 : NONE
# - classe 9 : Trochospiralé multiloculaire

mask[mask==6] = 0
mask[mask==7] = 0
mask[mask==8] = 0
mask[mask==9] = 6

A,B,C = mask.shape

print('shape image:', A,B,C)

path = '/media/instadeep/DATA/projects/cube_13B/mini_cube/dataset/'


idx = 0
for i in range(0,A-128,20):
    for j in range(0,B-128,60):
        for k in range(0,C-128,40):
            img_ = img[i:i+128, j:j+128, k:k+128]
            mask_ = mask[i:i+128, j:j+128, k:k+128]
            idx += 1
            print(path + 'img' + str(idx) + '.tif')
            imsave(path + 'train/image/img' + str(idx) + '.tif', img_)
            imsave(path + 'train/annotation/img' + str(idx) + '.tif', mask_)






# for ax in [(0,1), (0,2), (1,2)]:
#     for k in range(4):
#         for flp in ['n', 'v', 'h']:
#             img_ = np.rot90(img,k,ax)
#             mask_ = np.rot90(mask,k,ax)
#             if flp == 'v':
#                 for i in range(len(img)):
#                     img_[i] = np.flipud(img_[i])
#                     mask_[i] = np.flipud(mask_[i])
#             elif flp == 'h':
#                 for i in range(len(img)):
#                     img_[i] = np.fliplr(img_[i])
#                     mask_[i] = np.fliplr(mask_[i])
#             idx += 1
#             print(path + 'img' + str(idx) + '.tif')
#             imsave(path + 'img' + str(idx) + '.tif', img_)
#             imsave(path + 'msk' + str(idx) + '.tif', mask_)