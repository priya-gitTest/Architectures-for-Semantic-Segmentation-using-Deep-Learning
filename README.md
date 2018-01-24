# Architectures-for-Semantic-Segmentation-using-Deep-Learning

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Keras%20Implementation%20of%20Semantic%20Segmentation%20architectures&url=https://github.com/dalmia/Architectures-for-Semantic-Segmentation-using-Deep-Learning&hashtags=deeplearning,computervision,segmentation,machinelearning,keras)

[![apm](https://img.shields.io/apm/l/vim-mode.svg)]()
[![Build Status](https://travis-ci.org/athityakumar/colorls.svg?branch=master)](https://travis-ci.org/athityakumar/colorls)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields)](http://makeapullrequest.com)

This repository contains the implementations of [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) and [V-Net](https://arxiv.org/abs/1606.04797) in Keras using TensorFlow backend. Also, `data_prep.py` contains the code for random elastic deformations applied to the input images for data augmentation, which were specified to be of importance in both the papers.

## Data

The `data` folder contains pre-processed images, converted to .tif format from 3-D volume tiff as present in the [challenge website](http://brainiac2.mit.edu/isbi_challenge/).
## Architectures

- ### U-Net

![u-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

**Highlights**

- Encoder-decoder architecture as proposed in [Fully Convolutional Networks for Semantic Segmentation
](https://arxiv.org/abs/1605.06211)

- Each stage in the encoder part has 2 3x3 conv layers followed by a MaxPooling layer, which performs the downsampling operation.
- Number of filters double in each subsequent stage of the encoder part.
- In the decoder part, transpose convolution is used for upsampling and high-level features of the corresponding stage in the encoder part are concatenated at the beginning of each stage.
- The last layer uses 1x1 convolutions to give the final segmentation map.
- Extensive use of data augmentation, specially random elastic transformations.

- ### V-Net

![v-net](http://mattmacy.io/vnet.pytorch/images/diagram.png)

**Highlights**

- Similar architecture to U-Net, but with residual connections (https://arxiv.org/abs/1512.03385) between the input and output of each stage.

- Downsampling performed by strided convolution instead of MaxPooling.
- Varying number of convolutional layers in each stage.
- PReLU used as the activation. 

Although the original papers don't specify, I have always found **Batch Normalization** to speed up the training process and hence, the implementation includes that. Also, **Dropout** is applied between the convolutional layers.

## Running the model

First, prepare the augmented images by running:

```bash
python data_prep.py
```
Then, run the script corresponding to the architecture. E.g. for training U-Net, run the following:
```bash
python run_unet.py
```

## Dependencies

- TensorFlow
- Keras
- Numpy
- OpenCV

Install them using [pip](https://pypi.python.org/pypi/pip).

## Contributing
Feel free to create a Pull Request to add other semantic segmentation architectures and/or benchmark results or various datasets. If you are a beginner, you can refer [this](https://opensource.guide/how-to-contribute/) for getting started.

## Support
If you found this useful, please consider starring(★) the repo so that it can reach a broader audience.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/dalmia/Architectures-for-Semantic-Segmentation-using-Deep-Learning/blob/master/LICENSE) file for details.

## References

The pre-processed images and most part of `data_prep.py` were taken from [here](https://github.com/zhixuhao/unet).
