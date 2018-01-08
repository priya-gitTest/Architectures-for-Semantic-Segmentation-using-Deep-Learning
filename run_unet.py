from unet import *
from data_prep import *

data = dataProcess(512,512)
imgs_train, imgs_mask_train = data.load_train_data()
imgs_test = data.load_test_data()

unet = Unet(imgs_train, imgs_mask_train, imgs_test)
unet.train()
