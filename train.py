import os
import numpy as np
import datetime as dt
import torch
import torch.nn as nn
from torchvision import models
from load_data import *

PATH = './checkpoint'
PRE_TRAINED_MODEL = 'model.pth'
FINE_TUNED_MODEL = 'fine_tuned_model.pth'
PREDICT_MODEL = 'extra_model_epoch4.pth'

def load_model_optim(is_training = True, mymodel = True):
    # two steps to train and fine_tune a pre-trained model
    # when is_training is false, load fine_tuned model, whatever mymodel is True or False
    model = models.squeezenet1_1(pretrained = (not mymodel) and (is_training))
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2)
    model.num_classes = NCLS
    final_conv = nn.Conv2d(512, NCLS, kernel_size=1)
    model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

    # PyTorch torch.optim.SGD is SGD with momentum = 0.9
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),\
                                          lr = 0.01)

    if is_training:
        if not mymodel:
            # first train on PyTorch pre-trained model
            for param in model.parameters():
                param.requires_grad = False

            for param in model.features[0].parameters():
                param.requires_grad = True

            for param in final_conv.parameters():
                param.requires_grad = True

        else:
            # second train on my pre-trained model to fine tune
            if not os.path.exists(os.path.join(PATH, PRE_TRAINED_MODEL)):
                raise FileNotFoundError('can not find the model file.')

            checkpoint = torch.load(os.path.join(PATH, PRE_TRAINED_MODEL))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            for param in model.parameters():
                param.requires_grad = True
    else:
        # do not train, load model to predict
        if not os.path.exists(os.path.join(PATH, PREDICT_MODEL)):
            raise FileNotFoundError('can not find the model file.')

        checkpoint = torch.load(os.path.join(PATH, PREDICT_MODEL))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('loading model params which trained {} epochs, {} iterations.'.format(
                checkpoint['epoch'],
                checkpoint['iteration']
        ))

        # for param in model.parameters():
        #     param.requires_grad = False

        model.eval()

    return model, optimizer

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train():
    dtype = torch.float
    device = torch.device('cuda:0')
    # load model and optimizer params
    model, optimizer = load_model_optim(is_training = True, mymodel = True)

    # load model and optimizer params to GPU
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            # print (type(v))
            if torch.is_tensor(v):
                state[k] = v.to(device)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()
    # load data and training
    stroke2img = Strokes2Imgs(size = 224)
    dataset = Strokes_Imgs_data(chunk_size = 8 , transform = stroke2img)
    dataloader = data.DataLoader(dataset, batch_size = 8,
                                    num_workers = 8, collate_fn = collate_fn)
    # init
    loss_history = []
    errin_history = []
    sum_correct = 0
    sum_all = 0
    iter = 0
    # define max number of epoches
    num_epoches = 5

    # training...
    for epoch in range(num_epoches):
        start = dt.datetime.now()
        print('epoch {} start at time {}'.format(epoch + 1, start))
        running_loss = 0.0

        for d in dataloader:
            inputs = d['drawing'].to(device)
            labels = d['y'].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            # calculate err_in
            sum_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).float()
            sum_all += float(labels.size()[0])
            # define loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # tensor.item(), from tensor take value
            running_loss += loss.item()
            if iter % 1000 == 999:
                accuracy = sum_correct / sum_all
                print('[%d, %5d] loss: %.3f; acc: %.3f' %
                        (epoch + 1, iter + 1, running_loss / 1000, accuracy))
                loss_history.append(running_loss)
                sum_correct = 0
                sum_all = 0
                running_loss = 0.0
            # save model every 5000 iterations
            if iter % 5000 == 4999:
                if not os.path.isdir(PATH):
                    os.mkdir(PATH)

                torch.save({
                            'epoch': epoch + 1,
                            'iteration': iter + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, os.path.join(PATH, FINE_TUNED_MODEL))

                now = dt.datetime.now()
                print('saved model params at iter {}, {}'.format(iter + 1, now))

            # iter: 0 ~ (132812)*0.8 -> Train; else -> Val
            iter += 1

        # reduce learing rate every train epoch
        adjust_learning_rate(optimizer, decay_rate=.9)

        # save model every epoch one extre time
        if not os.path.isdir(PATH):
            os.mkdir(PATH)

        torch.save({
                    'epoch': epoch + 1,
                    'iteration': iter + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(PATH, 'extra_model_epoch{}.pth'.format(epoch + 1)))

        end = dt.datetime.now()
        print('epoch {} end at time {}, cost {}.'.format(epoch + 1, end, (end - start)))

def predict():
    dtype = torch.float
    device = torch.device('cuda:0')
    # load model and optimizer params
    model, _ = load_model_optim(is_training = True, mymodel = True)

    # load model and optimizer params to GPU
    model.to(device)

    stroke2img = Strokes2Imgs(size = 224)
    dataset = Strokes_Imgs_data(chunk_size = 8 , transform = stroke2img)
    dataloader = data.DataLoader(dataset, batch_size = 8,
                                    num_workers = 8, collate_fn = collate_fn)

    sum_correct_iter = 0
    sum_all_iter = 0
    sum_correct_epoch = 0
    sum_all_epoch = 0
    iter = 0

    for d in dataloader:
        inputs = d['drawing'].to(device)
        labels = d['y'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        # calculate err_in
        sorted, index = torch.sort(outputs, dim=1, descending=True)
        pred = index[:, 0:3] # take top-3 scores predict
        correct = float(np.sum([l in p for p, l in zip(pred, labels)]))
        sum = float(labels.size()[0])

        if iter == 0:
            print(correct, '\n' ,sum)

        sum_correct_iter += correct
        sum_all_iter += sum
        sum_correct_epoch += correct
        sum_all_epoch += sum

        if iter % 1000 == 999:
            accuracy_in = sum_correct_iter / sum_all_iter
            print('[%5d] acc: %.3f' %
                    (iter + 1, accuracy_in))
            sum_correct_iter = 0
            sum_all_iter = 0

        iter += 1    

def main():
    predict()

main()
