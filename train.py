import os
import numpy as np
import datetime as dt
import torch
from torchvision import models
from load_data import *

PATH = './checkpoint'
PRE_TRAINED_MODEL = 'first_step_model.pth'
SAVED_MODEL = 'first_step_model.pth'
BEST_VAL_MODEL = 'best_val_model_epoch.pth'
PREDICT_MODEL  = 'best_val_model_epoch.pth'

def load_model_optim(is_training = True, pretrained = True):
    # two steps to train and fine_tune a pre-trained model
    # pretrained = False means that using my pre-trained model, not PyTorch pre-trained model
    model = models.densenet121(pretrained = pretrained)
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier  = torch.nn.Linear(1024, NUM_CLASSES)

    # PyTorch torch.optim.SGD is SGD with momentum = 0.9
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),\
                                          lr = 0.01)

    if is_training:
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False

            for param in model.features[0].parameters():
                param.requires_grad = True

            for param in model.classifier.parameters():
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
        # for predict
        if not os.path.exists(os.path.join(PATH, PREDICT_MODEL)):
            raise FileNotFoundError('can not find the model file.')

        checkpoint = torch.load(os.path.join(PATH, PREDICT_MODEL))
        model.load_state_dict(checkpoint['model_state_dict'])

    return model, optimizer

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train():
    dtype = torch.float
    device = torch.device('cuda:0')
    # load model and optimizer params;
    # pretrained = False means that using my pre-trained model, not PyTorch pre-trained model
    model, optimizer = load_model_optim(is_training = True, pretrained = True)

    # load model to GPU
    model.to(device)
    # optimizer params to GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            # print (type(v))
            if torch.is_tensor(v):
                state[k] = v.to(device)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # load data and training
    stroke2img = Strokes2Imgs(size = 224)

    # init
    loss_history = []
    acc_in_history = []
    acc_out_history = []

    # define max number of epoches
    best_val_acc = 0.
    num_epoches = 5

    # training...
    for epoch in range(num_epoches):
        start = dt.datetime.now()
        print('epoch {} start at time {}'.format(epoch + 1, start))

        # train
        running_loss = 0.0
        sum_correct = 0
        sum_all = 0
        sum_correct_epoch = 0
        sum_all_epoch = 0
        iter = 0

        dataset = Strokes_Imgs_data(data_dir = TRAIN_DIR,
                     example_num = int(NROWS * NUM_CLASSES * 0.8),
                     chunk_size = 4 , transform = stroke2img)
        dataloader = data.DataLoader(dataset, batch_size = 8,
                                        num_workers = 8, collate_fn = collate_fn)
        model.train()
        for d in dataloader:
            inputs = d['drawing'].to(device)
            labels = d['y'].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            # calculate acc_in
            _, index = torch.sort(outputs, dim = 1, descending=True)
            pred = index[:, 0:3] # take top-3 scores predict
            correct_num = float(np.sum([l in p for p, l in zip(pred, labels)]))
            label_num = float(labels.size()[0])

            sum_correct += correct_num
            sum_all += label_num
            sum_correct_epoch += correct_num
            sum_all_epoch += label_num

            # define loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # tensor.item(), from tensor take value
            running_loss += loss.item()

            if iter % 200 == 199:
                accuracy = sum_correct / sum_all
                print('[%d, %5d] loss: %.3f; acc: %.3f' %
                        (epoch + 1, iter + 1, running_loss / 200, accuracy))
                loss_history.append(running_loss)
                sum_correct = 0
                sum_all = 0
                running_loss = 0.0

            # save model every 5000 iterations
            if iter % 1000 == 999:
                if not os.path.isdir(PATH):
                    os.mkdir(PATH)

                torch.save({
                            'epoch': epoch + 1,
                            'iteration': iter + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, os.path.join(PATH, SAVED_MODEL))

                now = dt.datetime.now()
                print('saved model params at iter {}, {}'.format(iter + 1, now))

            # iter: 0 ~ (132812)*0.8 -> Train; else -> Val
            iter += 1

        accuracy_in = sum_correct_epoch / sum_all_epoch
        acc_in_history.append(accuracy)
        print('epoch %d ------------------> accuracy_in: %.3f' % (epoch + 1, accuracy_in))

        # reduce learing rate every train epoch
        # adjust_learning_rate(optimizer, decay_rate=.9)

        # validation
        print('validation.........')
        correct_num_val = 0
        label_num_val = 0

        dataset = Strokes_Imgs_data(data_dir = VAL_DIR,
                     example_num = int(NROWS * NUM_CLASSES * 0.2),
                     chunk_size = 4 , transform = stroke2img)
        dataloader = data.DataLoader(dataset, batch_size = 8,
                                        num_workers = 8, collate_fn = collate_fn)
        model.eval()
        for d in dataloader:

            inputs = d['drawing'].to(device)
            labels = d['y'].to(device)

            with torch.no_grad():
                outputs = model(inputs)

            # calculate acc_in
            _, index = torch.sort(outputs, dim = 1, descending=True)
            pred = index[:, 0:3] # take top-3 scores predict
            correct_num = float(np.sum([l in p for p, l in zip(pred, labels)]))
            label_num = float(labels.size()[0])

            correct_num_val += correct_num
            label_num_val += label_num

        accuracy_out = correct_num_val / label_num_val
        acc_out_history.append(accuracy_out)
        print('epoch %d ------------------> accuracy_out: %.3f' % (epoch + 1, accuracy_out))

        # save model every epoch one extre time
        if (accuracy_out > best_val_acc):
            best_val_acc = accuracy_out

            if not os.path.isdir(PATH):
                os.mkdir(PATH)

            torch.save({
                        'epoch': epoch + 1,
                        'iteration': iter + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, os.path.join(PATH, BEST_VAL_MODEL))

        end = dt.datetime.now()
        print('epoch {} end at time {}, cost {}.'.format(epoch + 1, end, (end - start)))

def main():
    train()

if __name__ == '__main__':
    main()
