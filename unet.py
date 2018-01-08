from utils import *
from os.path import exists
from os import mkdir

import numpy as np

from keras.models import *
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard


class Unet(object):
    """
    The U-net architecture (https://arxiv.org/abs/1505.04597) inspired by the
    encoder-decoder architecture proposed in https://arxiv.org/abs/1605.06211.

    """
    def __init__(self, imgs_train, imgs_mask_train, imgs_test, batch_size=4, epochs=20, log_dir='./tb_logs/', checkpoint_dir='./weights', 
                 learning_rate=1e-4, keep_prob=0.5, img_rows = 128, img_cols = 160):
        """
        PARAMETERS
        ----------

        imgs_train: 4-D numpy array, dtype=float
                    Images to be used for training

        imgs_mask_train: 4-D numpy array, dtype=int
                         Masks corresponding to the training images

        imgs_test: 4-D numpy array, dtype=float
                   Images to be tested

        batch_size: int, default=4
                    Batch size used for training

        epochs: int, default=20
                Number of epochs to train the model

        log_dir: string,
                 location for tensorboard logs

        checkpoint_dir: string,
                        location for storing model checkpoints

        learning_rate: float, default=1e-4
                       Learning rate to be used for training

        keep_prob: float, default=0.5
                   Keep probability for Dropout

        img_rows: int, default=128
                  Height of each image

        img_cols: int, default=160
                  Width of each image
        """

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.imgs_train = imgs_train
        self.batch_size = batch_size
        self.epochs = epochs

        if not exists(log_dir):
            mkdir(log_dir)

        if not exists(checkpoint_dir):
            mkdir(checkpoint_dir)

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        self.self.learning_rate = learning_rate
        self.keep_prob = keep_probs
        self.imgs_mask_train = imgs_mask_train
        self.imgs_test = imgs_test


    def load_data(self):
        """
        RETURNS
        -------

        imgs_train: 4-D numpy array, dtype=float
                    Images to be used for training

        imgs_mask_train: 4-D numpy array, dtype=int
                         Masks corresponding to the training images

        imgs_test: 4-D numpy array, dtype=float
                   Images to be tested
        """

        return self.imgs_train, self.imgs_mask_train, self.imgs_test


    def convolutional_block(self, x, filters, kernel_size, stage=None, block=None, padding='valid'):
        """
        Performs a convolution followed by Batch normalization.

        PARAMETERS
        ----------
        x: tensor,
           Input to the block

        filters: int,
                 Number of filters for the convolution

        kernel_size: int or tuple,
                     Specifies the kernel size for the convolution

        stage: string, default=None
               The stage in the architecture where the convolution is applied

        block: string, default=None
               The block number within the stage specified above

        padding: {'valid', 'same'}, default='valid'
                 Padding for the convolution

        RETURNS
        -------

        x: tensor,
           Output of the (Conv + Batch Norm) operation

        """

        x = Conv2D(filters, kernel_size, padding = 'same', 
                   kernel_initializer = 'he_normal',
                   name='conv' + stage + '_' + block)(x)
        x = BatchNormalization(name='batch_norm' + stage + '_' + block)(x)
        x = Activation('relu', name='act' + stage + '_' + block)(x)
        return x


    def deconvolutional_block(self, x, filters, kernel_size=2, stage=None, block=None, 
                              strides=(2, 2)):

        """
        Performs a transpose convolution followed by Batch normalization.

        PARAMETERS
        ----------
        x: tensor,
           Input to the block

        filters: int,
                 Number of filters for the operation

        kernel_size: int or tuple,
                     Specifies the kernel size for the operation

        stage: string, default=None
               The stage in the architecture where the operation is applied

        block: string, default=None
               The block number within the stage specified above

        strides: int or tuple, default=(2, 2)
                 The stride to be used for the operation

        RETURNS
        -------

        x: tensor,
           Output of the (TransposeConv + Batch Norm) operation

        """

        x = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', 
                              kernel_initializer = 'he_normal',name='up' + stage)(x)
        x = BatchNormalization(name='batch_norm' + stage + '_' + block)(x)
        x = Activation('relu', name='act' + stage + '_' + block)(x)
        return x


    def get_unet(self):
        """
        Build the U-Net architecture.

        RETURNS
        -------

        model: Keras Model object,
               The U-Net model

        """

        inputs = Input((self.img_rows, self.img_cols,1))

        stage = '1'
        out1 = self.convolutional_block(inputs, base, 3, stage=stage, block='1', padding = 'same')
        out1 = self.convolutional_block(out1, base, 3, stage=stage, block='2', padding = 'same')

        pool1 = MaxPooling2D(pool_size=(2, 2))(out1)


        stage = '2'
        out2 = self.convolutional_block(pool1, base * 2, 3, stage=stage, block='1', padding = 'same')
        out2 = self.convolutional_block(out2, base * 2, 3, stage=stage, block='2', padding = 'same')

        drop2= Dropout(self.keep_prob, name='drop2')(out2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(drop2)


        stage = '3'
        out3 = self.convolutional_block(pool2, base * 4, 3, stage=stage, block='1', padding = 'same')
        out3 = self.convolutional_block(out3, base * 4, 3, stage=stage, block='2', padding = 'same')

        drop3 = Dropout(self.keep_prob, name='drop3')(out3) 
        pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(drop3)


        stage = '4'
        out4 = self.convolutional_block(pool3, base * 8, 3, stage=stage, block='1', padding = 'same')
        out4 = self.convolutional_block(out4, base * 8, 3, stage=stage, block='2', padding = 'same')

        drop4 = Dropout(self.keep_prob, name='drop4')(out4)    
        pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(drop4)


        stage = '5'
        out5 = self.convolutional_block(pool4, base * 16, 3, stage=stage, block='1', padding = 'same')
        out5 = self.convolutional_block(out5, base * 16, 3, stage=stage, block='2', padding = 'same')

        drop5 = Dropout(self.keep_prob, name='drop5')(out5)


        stage = '6'
        out6 = self.deconvolutional_block(drop5, base * 8, stage=stage, block='1')
        concatenate6 = concatenate([drop4, out6], axis = 3, name='concat6')

        out6 = self.convolutional_block(concatenate6, base * 8, 3, stage=stage, block='2', padding = 'same')
        out6 = self.convolutional_block(out6, base * 8, 3, stage=stage, block='3', padding = 'same')


        out7 =  self.deconvolutional_block(out6, base * 4, stage='7', block='1')
        concatenate7 = concatenate([drop3, out7], axis = 3, name='concat7')

        stage = '7'
        out7 = self.convolutional_block(concatenate7, base * 4, 3, stage=stage, block='2', padding = 'same')
        out7 = self.convolutional_block(out7, base * 4, 3, stage=stage, block='3', padding = 'same')

        out8 = self.deconvolutional_block(out7, base * 2, stage='8', block='1')

        concatenate8 = concatenate([drop2, out8], axis = 3, name='concat8')

        stage = '8'
        out8 = self.convolutional_block(concatenate8, base * 2, 3, stage=stage, block='2', padding = 'same')
        out8 = self.convolutional_block(out8, base * 2, 3, stage=stage, block='3', padding = 'same')


        stage = '9'
        out9 = self.deconvolutional_block(out8, base, stage=stage, block='1')
        concatenate9 = concatenate([out1, out9], axis = 3, name='concat9')

        out9 = self.convolutional_block(concatenate9, base, 3, stage=stage, block='2', padding = 'same')
        out9 = self.convolutional_block(out9, base, 3, stage=stage, block='3', padding = 'same')

        conv10 = Conv2D(1, 1, activation = 'sigmoid')(out9)

        model = Model(inputs=inputs, outputs = conv10)

        model.compile(optimizer = Adam(lr = self.learning_rate, beta_1 = 0.99), loss = dice_coef_loss, metrics = [dice_coef])

        return model


    def train(self):
        """
        Train the model

        """

        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print(model.summary())
        
        print("got unet")

        tb = TensorBoard(log_dir=self.log_dir, 
                         histogram_freq=0,
                         write_graph=True, 
                         write_images=False)

        checkpointer = ModelCheckpoint(join(self.checkpoint_dir, 'chkpts.{epoch:02d}-{val_loss:.2f}.hdf5'), 
                                       monitor='val_loss', 
                                       save_best_only=True, 
                                       save_weights_only=False, 
                                       verbose=1)

        print('Fitting model...')

        model.fit(imgs_train, imgs_mask_train,
                  batch_size=self.batch_size, 
                  epochs=self.epochs, 
                  verbose=1,
                  validation_split=0.2, 
                  shuffle=True, 
                  callbacks=[checkpointer, tb])

        return model

        print('Predicting test data')

        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save(join(result_save_root, 'imgs_mask_test.npy'), imgs_mask_test)

    def save_img(self):
        """
        Save the test predictions as images.
        
        """

        print("array to image")
        imgs = np.load(join(result_save_root, 'imgs_mask_test.npy'))
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save(join(result_save_root, "%d.jpg"%(i)))