from utils import *
from os.path import exists, join
from os import mkdir

import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, Cropping2D, BatchNormalization, Activation, PReLU, Add
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard

class Vnet(object):
    """
    The V-net architecture (https://arxiv.org/abs/1505.04597) inspired by the
    encoder-decoder architecture proposed in https://arxiv.org/abs/1605.06211,
    and uses residual connections (https://arxiv.org/abs/1512.03385) for speeding
    up the training process.

    """

    def __init__(self, imgs_train, imgs_mask_train, imgs_test, base=16, batch_size=4, epochs=20, log_dir='./tb_logs/', 
                 checkpoint_dir='./weights/', learning_rate=1e-4, keep_prob=0.5, img_rows = 512, img_cols = 512):
        """
        PARAMETERS
        ----------

        imgs_train: 4-D numpy array, dtype=float
                    Images to be used for training

        imgs_mask_train: 4-D numpy array, dtype=int
                         Masks corresponding to the training images

        imgs_test: 4-D numpy array, dtype=float
                   Images to be tested

        base: int, default=16
              The number of convolutional filters to use in the first stage

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

        img_rows: int, default=512
                  Height of each image

        img_cols: int, default=512
                  Width of each image

        """

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.imgs_train = imgs_train
        self.base = base
        self.batch_size = batch_size
        self.epochs = epochs

        if not exists(log_dir):
            mkdir(log_dir)

        if not exists(checkpoint_dir):
            mkdir(checkpoint_dir)

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
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


    def identity_transformed(self, x, filters, stage=None):
        """
        Transform the input so that it can be concatenate with the output
        of the convolutional layers.

        PARAMETERS
        ----------

        x: tensor,
           Input to be transformed.

        filters: int,
                 Number of filters in the output

        RETURNS
        -------

        x: tensor,
           The transformed input tensor

        """

        x = Conv2D(filters, 1, padding = 'same', kernel_initializer = 'he_normal',
                          name='conv_res' + stage)(x)
        x = BatchNormalization(name='batch_norm_res' + stage)(x)
        return x


    def convolutional_block(self, x, filters, kernel_size, stage, block, strides=(1, 1), activation=True, padding='valid'):
        """
        Convolutional Block consisting of BatchNormalization and PReLU as activation.

        PARAMETERS
        ----------

        x: input tensor
        filters: int,
                 Number of filters for the convolution.

        kernel_size: int or tuple of integers
                     Kernel size for the convolution

        stage: string,
               The stage in the architecture where the convolution is applied

        block: string,
               The block number within the stage specified above 

        strides: tuple, default=(1, 1)
                 The stride to be used for the convolution

        activation: bool, default=True
                    Whether or not to include the Activation Layer at the end.

        padding: {'same', 'valid'}, default='valid'
                 Same as that for Conv2D.

        RETURNS
        -------

        x: tensor,
           Output of the (Conv + Batch Norm) operation

        """

        x = Conv2D(filters, kernel_size, strides=strides, padding = 'same', 
                   kernel_initializer = 'he_normal',
                   name='conv' + stage + '_' + block)(x)
        x = BatchNormalization(name='batch_norm' + stage + '_' + block)(x)

        if activation:
            x = PReLU(name='act' + stage + '_' + block)(x)

        return x


    def deconvolutional_block(self, x, filters, kernel_size, stage=None, block=None, 
                              strides=(2, 2)):
        """
        Deconvolutional block consisting of BatchNormalization and PReLU as activation.

        PARAMETERS
        ----------

        x: input tensor
        filters: int,
                 Number of filters for the transpose convolution.

        kernel_size: int or tuple of integers
                     Kernel size for the transpose convolution

        stage: string,
               The stage in the architecture where the deconvolution is applied

        block: string,
               The block number within the stage specified above 

        strides: tuple, default=(2, 2)
                 The stride to be used for the transpose convolution

        RETURNS
        -------

        x: tensor,
           Output of the (TransposeConv + Batch Norm) operation

        """

        x = Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', 
                              kernel_initializer = 'he_normal',name='up' + stage)(x)
        x = BatchNormalization(name='batch_norm' + stage + '_' + block)(x)
        x = PReLU(name='act' + stage + '_' + block)(x)

        return x


    def get_vnet(self):
        """
        Build the V-Net architecture.

        RETURNS
        -------

        model: Keras Model object,
               The V-Net model

        """

        inputs = Input((self.img_rows, self.img_cols,1))

        # ----------------- Layer 1 -------------------------------

        stage = '1'
        in1 = self.identity_transformed(inputs, self.base, stage)

        out1 = self.convolutional_block(inputs, self.base, 5, stage=stage, block='1', activation=False, padding = 'same')
        add1 = Add(name='add1')([out1, in1])
        act1 = PReLU(name='act1_1')(add1)

        out1 = self.convolutional_block(act1, self.base, 2, stage=stage, block='2', padding = 'same', strides=(2, 2))

        # ----------------- Layer 2-------------------------------
        
        stage = '2'
        in2 = self.identity_transformed(out1, self.base * 2, stage)
        out2 = self.convolutional_block(out1, self.base * 2, 5, stage=stage, block='1', padding='same')
        out2 = self.convolutional_block(out2, self.base * 2, 5, stage=stage, block='2', activation=False, padding='same')
        add2 = Add(name='add2')([out2, in2])
        act2 = PReLU(name='act2_2')(add2)

        drop2= Dropout(self.keep_prob, name='drop2')(act2)
        out2 = self.convolutional_block(drop2, self.base * 2, 2, stage=stage, block='3', padding = 'same', strides=(2, 2))

        # ----------------- Layer 3 -------------------------------
        
        stage = '3'
        in3 = self.identity_transformed(out2, self.base * 4, stage)
        out3 = self.convolutional_block(out2, self.base * 4, 5, stage=stage, block='1', padding='same')
        out3 = self.convolutional_block(out3, self.base * 4, 5, stage=stage, block='2', padding='same')
        out3 = self.convolutional_block(out3, self.base * 4, 5, stage=stage, block='3', activation=False, padding='same')
        add3 = Add(name='add3')([out3, in3])
        act3 = PReLU(name='act3_3')(add3)
        
        drop3 = Dropout(self.keep_prob, name='drop3')(act3)
        out3 = self.convolutional_block(drop3, self.base * 2, 2, stage=stage, block='5', padding = 'same', strides=(2, 2))

        # ----------------- Layer 4 -------------------------------
        
        stage = '4'
        in4 = self.identity_transformed(out3, self.base * 8, stage)
        out4 = self.convolutional_block(out3, self.base * 8, 5, stage=stage, block='1', padding='same')
        out4 = self.convolutional_block(out4, self.base * 8, 5, stage=stage, block='2', padding='same')
        out4 = self.convolutional_block(out4, self.base * 8, 5, stage=stage, block='3', activation=False, padding='same')
        add4 = Add(name='add4')([out4, in4])
        act4 = PReLU(name='act4_3')(add4)            

        drop4 = Dropout(self.keep_prob, name='drop4')(act4)
        out4 = self.convolutional_block(drop4, self.base * 2, 2, stage=stage, block='4', padding = 'same', strides=(2, 2))

        # ----------------- Layer 5 -------------------------------
        
        stage = '5'
        in5 = self.identity_transformed(out4, self.base * 16, stage)
        out5 = self.convolutional_block(out4, self.base * 16, 5, stage=stage, block='1', padding='same')
        out5 = self.convolutional_block(out5, self.base * 16, 5, stage=stage, block='2', padding='same')
        out5 = self.convolutional_block(out5, self.base * 16, 5, stage=stage, block='3', activation=False, padding='same')

        add5 = Add(name='add5')([out5, in5])
        act5 = PReLU(name='act5_3')(add5)            

        drop5 = Dropout(self.keep_prob, name='drop5')(act5)

        # ----------------- Layer 6 -------------------------------
        
        stage = '6'
        out6 = self.deconvolutional_block(drop5, self.base * 8, 2, stage=stage, block='1')
        in6 = self.identity_transformed(out6, self.base * 16, stage)
        out6 = concatenate([drop4, out6], axis = 3, name='concat' + stage)
        out6 = self.convolutional_block(out6, self.base * 16, 5, stage=stage, block='2', padding='same')
        out6 = self.convolutional_block(out6, self.base * 16, 5, stage=stage, block='3', padding='same')
        out6 = self.convolutional_block(out6, self.base * 16, 5, stage=stage, block='4', activation=False, padding='same')
        add6 = Add(name='add6')([out6, in6])
        act6 = PReLU(name='act6_4')(add6)

        # ----------------- Layer 7 -------------------------------
        
        stage = '7'
        out7 = self.deconvolutional_block(act6, self.base * 4, 2, stage=stage, block='1')
        in7 = self.identity_transformed(out7, self.base * 8, stage)
        out7 = concatenate([drop3, out7], axis = 3, name='concat' + stage)
        out7 = self.convolutional_block(out7, self.base * 8, 5, stage=stage, block='2', padding='same')
        out7 = self.convolutional_block(out7, self.base * 8, 5, stage=stage, block='3', padding='same')
        out7 = self.convolutional_block(out7, self.base * 8, 5, stage=stage, block='4', activation=False, padding='same')
        add7 = Add(name='add7')([out7, in7])
        act7 = PReLU(name='act7_4')(add7)

        # ----------------- Layer 8 -------------------------------
        
        stage = '8'
        out8 = self.deconvolutional_block(act7, self.base * 2, 2, stage=stage, block='1')
        in8 = self.identity_transformed(out8, self.base * 4, stage)
        out8 = concatenate([drop2, out8], axis = 3, name='concat' + stage)
        out8 = self.convolutional_block(out8, self.base * 4, 5, stage=stage, block='2', padding='same')
        out8 = self.convolutional_block(out8, self.base * 4, 5, stage=stage, block='3', activation=False, padding='same')
        add8 = Add(name='add8')([out8, in8])
        act8 = PReLU(name='act8_3')(add8)

        # ----------------- Layer 9 -------------------------------
        
        stage = '9'
        out9 = self.deconvolutional_block(act8, self.base, 2, stage=stage, block='1')
        in9 = self.identity_transformed(out9, self.base * 2, stage)
        out9 = concatenate([act1, out9], axis = 3, name='concat' + stage)
        out9 = self.convolutional_block(out9, self.base * 2, 5, stage=stage, block='2', activation=False, padding='same')
        add9 = Add(name='add9')([out9, in9])
        act9 = PReLU(name='act9_2')(add9)

        conv10 = Conv2D(1, 1, kernel_initializer = 'he_normal', activation = 'sigmoid')(act9)

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
        model = self.get_vnet()
        print(model.summary())

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