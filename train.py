import os
import datetime as dt
import torch
import torch.nn as nn
from torchvision import models
from load_data import *

PATH = './checkpoint'
PRE_TRAINED_MODEL = 'model.pth'
FINE_TUNED_MODEL = 'fine_tuned_model.pth'

def load_model(is_training = True, mymodel = True):
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
            model.load_state_dict(torch.load(os.path.join(PATH, PRE_TRAINED_MODEL)))

            for param in model.parameters():
                param.requires_grad = True
    else:
        # do not train, load model to predict
        if not os.path.exists(os.path.join(PATH, FINE_TUNED_MODEL)):
            raise FileNotFoundError('can not find the model file.')
        model.load_state_dict(torch.load(os.path.join(PATH, FINE_TUNED_MODEL)))

        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    return model


def train():
    dtype = torch.float
    device = torch.device('cuda:0')
    model = load_model(is_training = True, mymodel = True)
    # model load to GPU
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),\
                                          lr=0.001)
    # load data and training
    stroke2img = Strokes2Imgs(size = 224)
    dataset = Strokes_Imgs_data(chunk_size = 8 , transform = stroke2img)
    dataloader = data.DataLoader(dataset, batch_size = 8,
                                    num_workers = 8, collate_fn = collate_fn)

    loss_history = []
    errin_history = []
    sum_correct = 0
    sum_all = 0
    num_epoches = 2
    num_iter = 0
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

            running_loss += loss.item()
            if num_iter % 1000 == 999:
                accuracy = sum_correct / sum_all
                print('[%d, %5d] loss: %.3f; acc: %.3f' %
                        (epoch + 1, num_iter + 1, running_loss / 1000, accuracy))
                loss_history.append(running_loss)
                sum_correct = 0
                sum_all = 0
                running_loss = 0.0

            if num_iter % 5000 == 4999:
                if not os.path.isdir(PATH):
                    os.mkdir(PATH)
                torch.save(model.state_dict(), os.path.join(PATH, FINE_TUNED_MODEL))
                now = dt.datetime.now()
                print('saved model params at iter {}, {}'.format(num_iter + 1, now))
            # num_iter: 0 ~ 31200 -> Train; else -> Val
            num_iter += 1

        end = dt.datetime.now()
        print('epoch {} end at time {}, cost {}.'.format(epoch + 1, end, (end - start)))

def main():
    train()

main()
