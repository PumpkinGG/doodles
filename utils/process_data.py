import os
import ast
import cv2
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm

TRAIN_DIR = './data/train'
VAL_DIR = './data/val'
NUM_CLASSES = 340
BASE_SIZE = 256
MIN_NUM_PER_FILE = 100000 # 101261
# Total train example num is: 8160122
# (not the number of kaggle given data, it's that I processed in shuffle_data.py)
N_TRAIN = 10000 * NUM_CLASSES
# Total val example num is: 2039878
# same as above
N_VAL = 500 * NUM_CLASSES


class simplified_data(data.Dataset):
    def __init__(self, mode = 'train',
                 transform = None):
        # save infor to class
        assert mode in ['train', 'valid']
        if mode == 'train':
            self.data_dir = TRAIN_DIR
            self.example_num = N_TRAIN
        elif mode == 'valid':
            self.data_dir = VAL_DIR
            self.example_num = N_VAL

        self.csv_file_list = os.listdir(self.data_dir)
        self.transform   = transform
        self.current_file_num  = 0
        self.current_file_iter = pd.read_csv(os.path.join(self.data_dir,
                                                self.csv_file_list[self.current_file_num]),
                                                compression = 'gzip', iterator = False)

    def __len__(self):
        # refer to shuffle_data.py, sample 25000 datas in every category
        # data.DataLoader will use this method to calculate epoch end iteration
        return self.example_num

    def __getitem__(self, idx):
        # first call this method
        if (idx // MIN_NUM_PER_FILE != self.current_file_num):
            self.current_file_num = idx // MIN_NUM_PER_FILE
            self.current_file_iter = pd.read_csv(os.path.join(self.data_dir,
                                                    self.csv_file_list[self.current_file_num]),
                                                    compression = 'gzip', iterator = False)

        # read example, index + 1 when finished
        idx = idx % MIN_NUM_PER_FILE
        temp = self.current_file_iter.iloc[idx, :]

        # transform strokes to an array of list; transform class to an array
        drawing = np.array(ast.literal_eval(temp['drawing']))
        y = np.array(temp['y'])

        sample = {'drawing': drawing,
                   'y': [y],
                   'cache': []}

        if self.transform:
            sample = self.transform(sample)

        return sample


###################################################################################
class Points2Imgs(object):
    def __init__(self, size = 256, thickness = 6, time_color = True, visual = False):
        # size: the size of bitmap image
        # thickness: the thickness of lines in bitmap
        # time_color: True for color changed with drawing time
        self.size = size
        self.thickness = thickness
        self.time_color = time_color
        self.visual = visual

    def __call__(self, sample):
        # sample: input, raw points lists
        drawing, y, cache = sample['drawing'], sample['y'], sample['cache']

        img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
        for t, stroke in enumerate(drawing): # strokes in one img
            for i in range(len(stroke[0]) - 1):
                color = 255 - min(t, 10) * 13 if self.time_color else 255
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                             (stroke[0][i + 1], stroke[1][i + 1]), color, self.thickness)
        if self.size != BASE_SIZE:
            img = cv2.resize(img, (self.size, self.size))

        if self.visual:
            cv2.imshow('result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        img = (img - 0) / 255 # normalization to 0 ~ 1
        img = (img - 0.5) / 0.5 # normalization to -1 ~ 1

        return {'drawing': torch.from_numpy(img).float().unsqueeze(0),
                'y': y,
                'cache': cache}

def null_imgs_collate(batch):
    # torch.stack changed to torch.cat
    drawing = [d['drawing'] for d in batch]
    truth = []
    cache = []
    input = torch.stack(drawing, dim = 0)

    if batch[0]['y'] != []:
        truth = [d['y'][0] for d in batch]
        truth = np.stack(truth, axis = 0)
        truth = torch.from_numpy(truth)

    if batch[0]['cache'] != []:
        cache = [d['cache'][0] for d in batch]
        cache = pd.DataFrame(cache)
        cache = cache.drop(['countrycode', 'drawing'], axis = 1)

    return input, truth, cache


###################################################################################
class Points2Strokes(object):
    def __init__(self, augment = False):
        self.augment = augment

    def __call__(self, sample):
        # sample: input, raw points lists
        drawing, y, cache = sample['drawing'], sample['y'], sample['cache']

        point = []
        for t,(x_,y_) in enumerate(drawing):
            point.append(np.array((x_,y_,np.full(len(x_),t)),np.float32).T)
        point = np.concatenate(point)

        stroke = self.point_to_stroke(point)

        # return {list, np.array}
        return {'drawing': stroke,
                'y': y,
                'cache': cache}

    def point_to_stroke(self, point):
        point = self.normalise_point(point)
        num_point = len(point)
        #stroke =[x,y,dt]
        #--------
        stroke = np.zeros((num_point,3),np.float32)
        stroke[:,2] = [1] + np.diff(stroke[:,2]).tolist()
        stroke[:,2] += 1
        # stroke[0] = [0,0,1]
        # stroke[1:] = point[1:] - point[:-1]

        return stroke

    def normalise_point(self, point):
        x_max = point[:,0].max()
        x_min = point[:,0].min()
        y_max = point[:,1].max()
        y_min = point[:,1].min()
        w = x_max-x_min
        h = y_max-y_min
        s = max(w,h)

        point[:,:2] = (point[:,:2]-[x_min,y_min])/s
        point[:,:2] = (point[:,:2]-[w/s*0.5,h/s*0.5])

        return point


###################################################################################
def null_stroke_collate(batch):
    # conbine chunks
    drawing = [d['drawing'] for d in batch]
    truth = []
    cache = []

    batch_size = len(drawing)
    #resort
    length  = np.array([len(drawing[b]) for b in range(batch_size)])
    argsort = np.argsort(-length)

    input = []
    for b in argsort:
        input.append(drawing[b])
        cache.append(cache[b])

    length = length[argsort]
    length_max = length.max()
    dim = len(drawing[b][0])

    pack = np.zeros((batch_size, length_max, dim), np.float32)
    for b in range(batch_size):
        pack[b, 0:length[b]] = input[b]
    input = torch.from_numpy(pack).float()

    if batch[0]['y'] != []:
        truth = [d['y'][0] for d in batch]
        truth = np.array(truth)
        truth = torch.from_numpy(truth).long()

    if batch[0]['cache'] != []:
        cache = [d['cache'][0] for d in batch]
        cache = pd.DataFrame(cache)

    return input, length, truth, cache


###################################################################################
def run_check_stroke():
    transform = Points2Strokes(augment = False)
    dataset = simplified_data(mode = 'train', transform = transform)
    dataloader = data.DataLoader(dataset, batch_size = 16,
                                    num_workers = 8, collate_fn = null_stroke_collate)
    iter = 0
    for input, length, truth, _ in dataloader:

        print(input.size())
        print(length)
        print(truth)

        iter += 1
        if iter == 2:
            return 0


###################################################################################
def run_check_img():
    # a example of how to use this model to load data
    transform = Points2Imgs(size = 224, visual = True)
    dataset = simplified_data(mode = 'train', transform = transform)
    dataloader = data.DataLoader(dataset, batch_size = 4,
                                    num_workers = 1, collate_fn = null_imgs_collate)

    num_epoches = 2
    num_iter = 0
    for epoch in range(num_epoches):
        for input, truth, _ in dataloader:

            num_iter += 1
            #print(epoch, num_iter)
            print(input.size())
            print(truth)
            if num_iter == 1:
                return 0

if __name__ == '__main__':
    run_check_stroke()
    # run_check_img()
