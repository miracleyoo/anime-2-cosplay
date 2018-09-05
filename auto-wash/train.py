# coding: utf-8
import torch
import torch.nn as nn
import torch.autograd
import os
import json
import datetime
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter


def training(opt, train_loader, test_loader, net, class_names):
    top_num= opt.TOP_NUM

    criterion = nn.BCEWithLogitsLoss(size_average=False)
    NUM_TRAIN_PER_EPOCH = len(train_loader)

    print('==> Loading Model ...')
    prefix = opt.NET_SAVE_PATH + opt.DATASET_PATH + '/'
    if not os.path.exists(prefix): os.mkdir(prefix)
    temp_model_name = prefix + '%s_model_temp.pkl' % net.__class__.__name__
    model_name =prefix + '%s_model.pkl' % net.__class__.__name__
    if os.path.exists(temp_model_name) and not opt.RE_TRAIN:
        net = torch.load(temp_model_name)
        print("Load existing model: %s" % temp_model_name)

    if opt.USE_CUDA: net.cuda()
    writer = SummaryWriter(opt.SUMMARY_PATH)

    optimizer = torch.optim.Adam(list(net.fc.parameters()), lr=opt.LEARNING_RATE)

    best_test_acc = 0
    for epoch in range(opt.NUM_EPOCHS):
        train_loss = 0
        train_acc = 0

        # Start training
        net.train()

        print('==> Preparing Data ...')
        for i, data in tqdm(enumerate(train_loader), desc="Training", total=NUM_TRAIN_PER_EPOCH, leave=False, unit='b'):
            inputs, labels, *_ = data
            if opt.USE_CUDA:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Do statistics for training
            train_loss += loss.data[0]
            predicts = torch.sort(outputs, descending=True)[1][:, :top_num]
            predicts = predicts.data

            num_correct = 0

            # Turn variables into normal list
            if opt.USE_CUDA:
                labels_data = labels.cpu().data.numpy()
                predicts    = predicts.cpu().tolist()
            else:
                labels_data = labels.data.numpy()
                predicts = predicts.tolist()

            for i, predict in enumerate(predicts):
                for label in predict:
                    if label in list(np.where(labels_data[i] == 1)[0]):
                        num_correct += 1
                        break

            train_acc += num_correct

        # Save a temp model
        torch.save(net, temp_model_name)

        # Start testing
        test_loss, test_acc, *_ = testing(opt, test_loader, net, class_names)

        # Output results
        writer.add_scalar("Train/loss", train_loss / opt.NUM_TRAIN, epoch)
        writer.add_scalar("Train/acc", float(train_acc) / opt.NUM_TRAIN, epoch)
        writer.add_scalar("Test/loss", test_loss / opt.NUM_TEST, epoch)
        writer.add_scalar("Test/acc", float(test_acc) / opt.NUM_TEST, epoch)

        print(
            'Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f, '
            % (epoch + 1, opt.NUM_EPOCHS,
               train_loss / opt.NUM_TRAIN, train_acc / opt.NUM_TRAIN,
               test_loss / opt.NUM_TEST, test_acc / opt.NUM_TEST))
        if (test_acc / opt.NUM_TEST) > best_test_acc:
            best_test_acc = test_acc / opt.NUM_TEST
            torch.save(net, model_name)

    print('==> Training Finished.')
    return net


def testing(opt, test_loader, net, class_names):
    net.eval()
    # top_num = opt.TOP_NUM
    test_loss = 0
    test_acc  = 0
    bad_case  = []
    non_normal_pics = []
    non_normal = class_names.index('normal')
    criterion = nn.BCEWithLogitsLoss(size_average=False)

    for i, data in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader), leave=False, unit='b'):
        inputs, labels, data_path, *_ = data
        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Compute the outputs and judge correct
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        predicts = torch.sort(outputs, descending=True)[1][:, 0]
        predicts = predicts.data
        num_correct = 0

        if opt.USE_CUDA:
            labels_data = labels.cpu().data.numpy()
            predicts = predicts.cpu().tolist()
        else:
            labels_data = labels.data.numpy()
            predicts = predicts.tolist()

        for i, predict in enumerate(predicts):
            if predict == non_normal:
                non_normal_pics.append(data_path[i])
            if predict in list(np.where(labels_data[i]==1)[0]):
                num_correct += 1
            else:
                bad_case.append(data_path[i])

        # Do statistics for training
        test_loss += loss.data[0]
        test_acc += num_correct

    return test_loss, test_acc, bad_case, non_normal


def output_vector(opt, net, data):
    net.eval()
    inputs, labels, *_ = data
    if opt.USE_CUDA:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    outputs = net(inputs)
    return outputs
