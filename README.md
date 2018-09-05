# Animation image to cosplay image Tool by Deep Convolution Generative Adversarial Networks

# Introduction 

Do you want to turn your favorite character's **anime images to his/her cosplay images**? Or do you want to do the opposit, to turn **cosplay images into anime** and more kawaii ones? If your answer is yes, you can continue your reading and try it youself!

This program is built to turn a certain character's anime image into a cosplay image using deep convolution generative adversarial network, or **GAN**. 

# Preparation

You need to search for some sample images in anime and cosplay for the character you want to train. Recommended anime images download source: [Danbooru](https://danbooru.donmai.us/) and you can use the following downloader to do this work: [gallery-dl](https://github.com/mikf/gallery-dl) As for cosplay images source, you can choose [banciyuan](https://bcy.net), which has highest quality cosplay pictures and is easy to search, or just simply use baidu or google, whose quality is lower but easier to download. You can utilize the baidu-spider in the project file.

You ought to remove the low quality pictures. Certainly, you can do it on hand, however, if you'd like to train another NN to help you do this work, you can go to [auto-wash-images](https://github.com/miracleyoo/auto-wash-images), to remove unpleasing and useless images from dataset automatically after manually give the program some example to learn.

During your preparation of dataset, there must be a large difficulty to do a series of operation to your downloaded images, such as rename, seperate into training and testing set, check if the image is broken... Then, you can utilize the **utils.py** file to help you.

 GPU server is necessary since to train a GAN network well is not a easy task. Also, more data you get, better result you may generate. 

# Brief theory

The program will initiate a anime image dataloader and a cosplay dataloader at the same time, and define a Generator together with Discriminator network. In the generator network, anime images will be condensed into a 32 x 4 x 4 feature map, and then it will be over-sampled by ConvTranspose2d again into a 3 x 64 x 64 cosplay image. See the mechanism more detailedly in the program.

# Mechanism

* After every 6 training iterations, the files `real_samples.png` and `fake_samples.png` are written to disk. With the samples from the generative model.

* After every 50 epoch, models are saved to: `netG_epoch_%d.pth` and `netD_epoch_%d.pth`
* All of the output above will be saved in `./outputs/`

# Acknowledgement

* This example implements the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)
* This project is based on the tutorial of pytorch, [dcgan](https://github.com/pytorch/examples/tree/master/dcgan)

## Usage
```
usage: main.py [-h] --dataset DATASET --dataroot DATAROOT [--workers WORKERS]
               [--batchSize BATCHSIZE] [--imageSize IMAGESIZE] [--nz NZ]
               [--ngf NGF] [--ndf NDF] [--niter NITER] [--lr LR]
               [--beta1 BETA1] [--cuda] [--ngpu NGPU] [--netG NETG]
               [--netD NETD]

optional arguments:
  -h, --help            show this help message and exit
  --dataroot_ani DATAROOT_ANI   path to anime dataset
  --dataroot_cos DATAROOT_COS   path to cosplay dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF
  --ndf NDF
  --niter NITER         number of epochs to train for
  --lr LR               learning rate, default=0.0001
  --beta1 BETA1         beta1 for adam. default=0.5
  --cuda                enables cuda
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
```
