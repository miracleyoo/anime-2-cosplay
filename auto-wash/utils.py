# coding: utf-8
import os
import math
import random
import shutil
import numpy as np
import time
import re
import pickle
import warnings

from PIL import Image


warnings.filterwarnings("ignore")


def load_data(root='./Datasets/'):
    """
    :Outputs:
        train_pairs : the path of the train  images and their labels' index list
        test_pairs  : the path of the test   images and their labels' index list
        class_names : the list of classes' names
    :param root : the root location of the dataset.
    """
    if not root.endswith('/'):
        root=root+'/'
    IMG_PATH  = [root + 'train_data/', root+ 'test_data/']
    NAME_PATH = './source/reference/names.pkl'

    class_names = pickle.load(open(NAME_PATH, 'rb'))

    # Cope with train data
    files = []
    train_dirs = [IMG_PATH[0] + i + "/" for i in next(os.walk(IMG_PATH[0]))[1]]
    [files.extend([i + j for j in next(os.walk(i))[2] if "jpg" in j]) for i in train_dirs]
    train_pairs = [(i, class_names.index(i.split('/')[-2])) for i in files]

    # Cope with test data
    files = []
    test_dirs = [IMG_PATH[1] + i + "/" for i in next(os.walk(IMG_PATH[1]))[1]]
    [files.extend([i + j for j in next(os.walk(i))[2] if "jpg" in j]) for i in test_dirs]
    test_pairs = [(i, class_names.index(i.split('/')[-2])) for i in files]

    return train_pairs, test_pairs, class_names


def sep_data(opt):
    """
    When there is only a train_data folder, this function will divide these images randomly in to
    train_data and test_data, while the ration is determined by a parameter defined in opt.
    :param opt: the config object of this project.
    :return: None
    """
    train_path = './Datasets/'+opt.DATASET_PATH+'train_data/'
    dirs = [train_path + i + "/" for i in next(os.walk(train_path))[1]]

    for dirn in dirs:
        files = [dirn + file for file in next(os.walk(dirn))[2] if "jpg" in file]
        file_num = len(files)
        if file_num>0:
            test_part = random.sample(files, math.floor(file_num*(1-opt.TRAINDATARATIO)))
            test_path = './Datasets/'+ opt.DATASET_PATH + 'test_data/'+ dirn.split('/')[-2]+'/'
            if not os.path.exists(test_path): os.mkdir(test_path)
            for file in test_part:
                shutil.move(file, test_path)

            names = [test_path + name for name in os.listdir(test_path) if not name.startswith('.')]
            names.sort(key=lambda x:int(os.path.splitext(os.path.basename(x))[0]))
            for i in range(len(names)):
                os.rename(names[i], test_path + str(i) + '.jpg')

            names = [dirn + name for name in os.listdir(dirn) if not name.startswith('.')]
            names.sort(key=lambda x:int(os.path.splitext(os.path.basename(x))[0]))
            for i in range(len(names)):
                os.rename(names[i], dirn + str(i) + '.jpg')


def check_img(path):
    """
    Check all of the images whether there is something wrong and it cannot be opened.
    Also, if a image has less than 3 channels or it is not a jpg file, it will be removed.
    :param path:
    :return:None
    """
    # path = path + opt.DATASET_PATH + 'train_data/'
    dirs = [path + i + "/" for i in next(os.walk(path))[1]]
    for dirn in dirs:
        files = [dirn + file for file in next(os.walk(dirn))[2] if "jpg" in file]
        file_num = len(files)
        if file_num>0:
            for filen in files:
                try:
                    img = Image.open(filen)
                    imgnp = np.array(img)
                    filename, suffix = os.path.splitext(filen)
                    if suffix=='.gif':
                        os.remove(filen)
                    elif len(imgnp.shape)!=3 or imgnp.shape[0] < 3:
                        os.remove(filen)
                    elif suffix != '.jpg':
                        img.save(filename+'.jpg')
                        os.remove(filen)
                except OSError:
                    os.remove(filen)
        rename_folder(dirn)


def coalesce_dirs(path):
    """
    This function aims to extract all of the jpg image files, rename them, and then move them to the root path.
    It must meet the requirement that there are only dirs in this path and there can only be images with the suffix jpg.
    :param path: The folder you want to coalesce.
    :return: None
    """
    if not path.endswith('/'):
        path=path+'/'
    dirs = [path + i + "/" for i in next(os.walk(path))[1]]
    for i, dirn in enumerate(dirs):
        files = [dirn + file for file in next(os.walk(dirn))[2] if "jpg" in file]
        if len(files) > 0:
            rename_folder(dirn, prefix=str(i)+'_')
    for i, dirn in enumerate(dirs):
        files = [dirn + file for file in next(os.walk(dirn))[2] if "jpg" in file]
        if len(files) > 0:
            for filen in files:
                shutil.move(filen, path +filen.split('/')[-1])
            shutil.rmtree(dirn)
    rename_folder(path)


def re_sep(opt, root = './Datasets/'):
    """
    This function aims to redo the separation of dataset images after some arrangement of it like data-washing.
    :param opt:the config object of this project.
    :param root:the place where train_dataset and test_dataset are stored.
    :return:None
    """
    if not root.endswith('/'):
        root=root+'/'
    train_path=root+'train_data/'
    test_path =root+'test_data/'
    train_dirs = [train_path + i + "/" for i in next(os.walk(train_path))[1]]
    test_dirs  = [test_path + i + "/" for i in next(os.walk(test_path))[1]]
    for dirn in train_dirs:
        rename_folder(dirn,prefix='0_')
    for dirn in test_dirs:
        rename_folder(dirn)
        files = [dirn + file for file in next(os.walk(dirn))[2] if "jpg" in file]
        if len(files) > 0:
            for filen in files:
                shutil.move(filen, train_path+filen.split('/')[-2]+'/'+filen.split('/')[-1])
    for dirn in train_dirs:
        rename_folder(dirn,prefix='')
    sep_data(opt)


def rename_folder(path, prefix=''):
    """
    Rename all of the image files in a certain path in the format "`prefix`_i.jpg"
    :param path: the path in which the process is implemented.
    :param prefix: the prefix you want to add to all of the files' new name.
    :return: None
    """
    # pattern = re.compile('\d*')
    if not path.endswith('/'):
        path=path+'/'
    files = [path + name for name in os.listdir(path) if not name.startswith('.')]
    for i in range(len(files)):
        os.rename(files[i], path + 'temp_' + str(i) + '.jpg')
    for i in range(len(files)):
        os.rename(path + 'temp_' + str(i) + '.jpg', path + prefix + str(i) + '.jpg')
    # files.sort(key=lambda x:int(pattern.match(os.path.basename(x)).group(0)))
    # for i in range(len(files)):
    #     os.rename(files[i], path + prefix + str(i) + '.jpg')


def gen_name(opt, path = './Datasets/', out_path='./source/reference/names.pkl'):
    """
    Generate a file which contains all of the categories the dataset has.
    :return:
    """
    path = path + opt.DATASET_PATH + '/'
    dirs = []
    dataset_pathes = [path + i + "/" for i in next(os.walk(path))[1]]
    for i, in_path in enumerate(dataset_pathes):
        dirs.extend([i.strip('/') for i in next(os.walk(in_path))[1]])
    classes = list(set(dirs))
    # for i in range(len(dirs)-1):
    #     for j in range(classes):
    #         if classes[j] not in dirs[i+1]:
    #             list.remove(classes, classes[j])
    pickle.dump(classes, open(out_path, 'wb'))


def folder_init(opt):
    """
    Initialize folders required
    """
    if not os.path.exists('source'): os.mkdir('source')
    if not os.path.exists('source/bad_case'): os.mkdir('source/bad_case')    
    if not os.path.exists('source/reference'): os.mkdir('source/reference')
    if not os.path.exists('source/summary'): os.mkdir('source/summary')
    if not os.path.exists(opt.NET_SAVE_PATH): os.mkdir(opt.NET_SAVE_PATH)
    if not os.path.exists(opt.NET_SAVE_PATH + opt.DATASET_PATH): os.mkdir(opt.NET_SAVE_PATH + opt.DATASET_PATH)
