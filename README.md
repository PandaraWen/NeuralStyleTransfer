# NeuralStyleTransfer

Source code referred in [blog](https://pandara.xyz/2018/06/09/nst/). This project is one of assignments of [Deep Learning Specialization][coursera_deeplearning].

## Dependence

* Python 3
* scipy, matplotlib, numpy, tensorflow

## How to use

1. Run `mkdir resources`;
2. Download VGG-19 pretrained model from [here][vgg19_download_link] and put it into `./resource`;
3. Put your `content.jpg` and `style.jpg` into `./resource`, **make sure they are in the same dimensions**;
4. Run `python3 main.py`;


[vgg19_download_link]: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
[coursera_deeplearning]: https://www.coursera.org/specializations/deep-learning

