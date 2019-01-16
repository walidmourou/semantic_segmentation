from data_gen import Data_Generator_from_folder, Data_Generator_from_cube

from model import UNet_keras

# Image specs
IMG_WIDTH = 800
IMG_HEIGHT = 800
IMG_CHANNELS = 1
N_CLASS = 5
BATCH_SIZE = 4 #*3
NB_EPOCHS = 500
LR=0.001

# Create Image Iterator from Cube
IMAGE_PATH_TRAIN = '/media/instadeep/DATA/projects/cube_13B/BU_13B_x1600_y1600_z1600.tif'
ANNO_PATH_TRAIN = '/media/instadeep/DATA/projects/cube_13B/masks_GT.raw'

training_generator = Data_Generator_from_cube(IMAGE_PATH_TRAIN, ANNO_PATH_TRAIN, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, N_CLASS, convert_1class=False)
validation_generator = Data_Generator_from_cube(IMAGE_PATH_TRAIN, ANNO_PATH_TRAIN, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, N_CLASS, convert_1class=False)

# # Create image iterator from folder
# IMAGE_PATH_TRAIN = '/media/instadeep/DATA/projects/cube_13B/train/image/'
# ANNO_PATH_TRAIN = '/media/instadeep/DATA/projects/cube_13B/train/annotation/'

# IMAGE_PATH_VAL = '/media/instadeep/DATA/projects/cube_13B/validation/image/'
# ANNO_PATH_VAL = '/media/instadeep/DATA/projects/cube_13B/validation/annotation/'

# training_generator = Data_Generator_from_folder(IMAGE_PATH_TRAIN, ANNO_PATH_TRAIN, BATCH_SIZE,
#                                                  IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, N_CLASS)

# validation_generator = Data_Generator_from_folder(IMAGE_PATH_VAL, ANNO_PATH_VAL, BATCH_SIZE,
#                                                    IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, N_CLASS)

class_unet = UNet_keras(IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, N_CLASS=N_CLASS)

model = class_unet.graph(IN_FILTERS = 32, KERNEL_INIT = 'he_normal', VERBOSE = False)

# Training
class_unet.train(model, training_generator, validation_generator, BATCH_SIZE=BATCH_SIZE,
                         NB_EPOCHS=NB_EPOCHS, LR=LR, loss='dice_coe', num_gpu=[2])


# # Inference
# ckpt_path = '/media/instadeep/DATA/projects/semantic-segmentation/UNet_keras/unet_2018_07_08_13_13_44.h5'
# image_path = '/media/instadeep/DATA/projects/cube_13B/BU_13B_x1600_y1600_z1600.tif'
# mask_path = '/media/instadeep/DATA/projects/cube_13B/masks_GT.raw'

# class_unet = UNet_keras(IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS, N_CLASS=N_CLASS)

# model = class_unet.graph(IN_FILTERS = 32, KERNEL_INIT = 'he_normal', VERBOSE = False, weights=ckpt_path)

# class_unet.inference(model, ckpt_path, image_path, mask_path)