# coding: utf-8
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Config(object):
    def __init__(self):
        self.USE_CUDA            = torch.cuda.is_available()
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.DATASET_PATH        = "cos-ani"
        self.SUMMARY_PATH        = "./source/summary/" + self.DATASET_PATH + '/'
        self.TRAINDATARATIO      = 0.7
        self.RE_TRAIN            = False
        self.IS_TRAIN            = True
        self.PIC_SIZE            = 256
        self.NUM_TEST            = 0
        self.NUM_TRAIN           = 0
        self.TOP_NUM             = 1
        self.NUM_EPOCHS          = 50
        self.BATCH_SIZE          = 1
        self.TEST_BATCH_SIZE     = 8
        self.NUM_WORKERS         = 4
        self.NUM_CLASSES         = 8
        self.LEARNING_RATE       = 0.001
