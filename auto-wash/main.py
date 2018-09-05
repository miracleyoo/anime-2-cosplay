# coding: utf-8
import torchvision.transforms as transforms
import torchvision.models as models
import sys
import pickle

from torch.utils.data import DataLoader
from utils import *
from data_loader import *
from train import *
from config import Config

opt = Config()

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((opt.PIC_SIZE, opt.PIC_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((opt.PIC_SIZE, opt.PIC_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

folder_init(opt)
gen_name(opt)
train_pairs, test_pairs, class_names = load_data("./Datasets/"+opt.DATASET_PATH+ '/')

opt.NUM_CLASSES = len(class_names)

net = models.resnet152(pretrained=True)
fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features, opt.NUM_CLASSES)

if opt.IS_TRAIN:
    trainDataset = COS_DES(train_pairs, opt, transform_train)
    train_loader = DataLoader(dataset=trainDataset, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=opt.NUM_WORKERS, drop_last=False)

    testDataset  = COS_DES(test_pairs, opt, transform_test)
    test_loader  = DataLoader(dataset=testDataset,  batch_size=opt.TEST_BATCH_SIZE, shuffle=False, num_workers=opt.NUM_WORKERS, drop_last=False)

    opt.NUM_TRAIN    = len(trainDataset)
    opt.NUM_TEST     = len(testDataset)

    net = training(opt, train_loader, test_loader, net, class_names)

else:
    test_pairs.extend(train_pairs)
    testDataset  = COS_DES(test_pairs, opt, transform_test)
    test_loader  = DataLoader(dataset=testDataset,  batch_size=opt.TEST_BATCH_SIZE, shuffle=False, num_workers=opt.NUM_WORKERS, drop_last=False)
    model_name = opt.NET_SAVE_PATH + opt.DATASET_PATH + '%s_model.pkl' % net.__class__.__name__
    if os.path.exists(model_name):
        net = torch.load(model_name)
        print("Load existing model: %s" % model_name)
        test_loss, test_acc, bad_case, non_normal = testing(opt, test_loader, net, class_names)
        pickle.dump(bad_case, open('./source/bad_case/%s_bad_case.pkl'%opt.DATASET_PATH, 'wb'))
    else:
        try:
            sys.exit(0)
        except:
            print("Error!You haven't trained your net while you try to test it.")
        finally:
            print('Program stopped.')
        

