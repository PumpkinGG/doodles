import os
import torch
import torch.utils.data as data
import datetime as dt

import nets
import utils

MODEL_PATH = './checkpoint/'
SAVE_MODEL = 'rnn_pretrained.pth'
PRE_TRAIN_MODEL = 'rnn_pretrained.pth'

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train_cnn(pre_trained = False):
    # define using gpu or cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    # define network
    model = nets.MobileNetV2(utils.NUM_CLASSES)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=0.01, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                                 lr=0.001)
    criterion = nets.softmax_cross_entropy_criterion
    max_precision = 0.
    # load model params if define pre_trained True
    if pre_trained:
        checkpoint = torch.load(os.path.join(MODEL_PATH, PRE_TRAIN_MODEL))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        max_precision = checkpoint['precision']

    # load model and optimizer params to cuda/GPU
    model = model.to(device)
    for param_group in optimizer.param_groups:
        for k, v in param_group.items():
            if torch.is_tensor(v):
                param_group[k] = v.to(device)

    max_epoch = 20

    print('         loss  | prec      top      ')
    print('[ iter ]       |           1  ... k ')
    print('----------------------------------')

    for epoch in range(max_epoch):
        start = dt.datetime.now()
        print('# training......................')
        transform = utils.Points2Imgs(size = 224)
        dataset = utils.simplified_data(mode = 'train', transform = transform)
        dataloader = data.DataLoader(dataset, batch_size = 32,
                                        num_workers = 8, collate_fn = utils.null_imgs_collate)
        model.train()
        i = 0
        optimizer.zero_grad()
        for input, truth in dataloader:

            input = input.to(device)
            truth = truth.to(device)

            logit = model(input)
            loss  = criterion(logit, truth)
            precision, top = nets.metric(logit, truth)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            if i%200==0:
                print('[%06d] %0.3f | ( %0.3f ) %0.3f  %0.3f'%(
                    i, loss.item(),precision.item(), top[0].item(),top[-1].item(),
                ))
            i = i+1

        print('# validating......................')
        transform = utils.Points2Imgs(size = 224)
        dataset = utils.simplified_data(mode = 'valid', transform = transform)
        dataloader = data.DataLoader(dataset, batch_size = 32,
                                        num_workers = 8, collate_fn = utils.null_imgs_collate)
        model.eval()
        i = 0
        average_precision = 0
        for input, truth in dataloader:

            input = input.to(device)
            truth = truth.to(device)

            with torch.no_grad():
                logit = model(input)

            loss  = criterion(logit, truth)
            precision, top = nets.metric(logit, truth)

            if i%200==0:
                print('[%06d] %0.3f | ( %0.3f ) %0.3f  %0.3f'%(
                    i, loss.item(),precision.item(), top[0].item(),top[-1].item(),
                ))
            i = i+1

        average_precision /= i
        if average_precision > max_precision:
            max_precision = average_precision
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'precision': average_precision,
                        }, os.path.join(MODEL_PATH, SAVE_MODEL))

        end = dt.datetime.now()
        print('# epoch {} over, average_precision is {}, cost {}'.format(epoch, average_precision, end - start))

if __name__ == '__main__':
    train_cnn(pre_trained = False)