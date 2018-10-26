import os
import ast
import cv2
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as trans
from tqdm import tqdm

TRAIN_DIR = './data/train'
VAL_DIR = './data/val'
NROWS = 5000 # do not greater than 10000
NUM_CLASSES = 340
BASE_SIZE = 256

TEST = False

class Strokes_Imgs_data(data.Dataset):
    def __init__(self, data_dir = TRAIN_DIR,
                 example_num = int(NROWS * NUM_CLASSES * 0.8),
                 chunk_size = 1,
                 transform=None):
        # save infor to class
        self.data_dir = data_dir
        self.csv_file_list = os.listdir(data_dir)
        self.csv_file_num  = len(self.csv_file_list)
        self.example_num = example_num
        self.chunk_size  = chunk_size
        self.transform   = transform
        self.current_file_iter = None
        self.current_file_num  = 0

    def __len__(self):
        # refer to shuffle_data.py, sample 25000 datas in every category
        # data.DataLoader will use this method to calculate epoch end iteration
        return (self.example_num // self.chunk_size)

    def __getitem__(self, idx):
        # first call this method
        if not self.current_file_iter:
            self.current_file_num = 0
            self.current_file_iter = pd.read_csv(os.path.join(self.data_dir,
                                                    self.csv_file_list[self.current_file_num]),
                                                    compression = 'gzip', iterator = True,
                                                    usecols = ['drawing', 'y'])
        # when raise StopIteration exception, load next csv file
        try:
            temp = self.current_file_iter.get_chunk(self.chunk_size)
        except StopIteration:
            self.current_file_num += 1
            if self.current_file_num == self.csv_file_num:
                self.current_file_num = 0
                self.current_file_iter = None
                raise StopIteration("All csv files had been loaded one time!")
            self.current_file_iter = pd.read_csv(os.path.join(self.data_dir,
                                                    self.csv_file_list[self.current_file_num]),
                                                    compression = 'gzip', iterator = True,
                                                    usecols = ['drawing', 'y'])
            temp = self.current_file_iter.get_chunk(self.chunk_size)

        # transform strokes to an array of list; transform class to an array
        drawing = np.array(temp['drawing'].apply(ast.literal_eval))
        cls = np.array(temp['y'])

        sample = {'drawing': drawing,
                   'y': torch.from_numpy(cls)}

        if self.transform:
            sample['drawing'] = self.transform(sample)

        return sample

class Strokes2Imgs(object):
    def __init__(self, size = 256, thickness = 6, time_color = True):
        # size: the size of bitmap image
        # thickness: the thickness of lines in bitmap
        # time_color: True for color changed with drawing time
        self.size = size
        self.thickness = thickness
        self.time_color = time_color

    def __call__(self, sample):
        # sample: input, raw strokes lists
        drawing = sample['drawing']
        img_batch = np.zeros((drawing.shape[0], self.size, self.size), np.uint8)

        for idx, dr in enumerate(drawing): # imgs in one chunk
            img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
            for t, stroke in enumerate(dr): # strokes in one img
                for i in range(len(stroke[0]) - 1):
                    color = 255 - min(t, 15) * 15 if self.time_color else 255
                    _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                                 (stroke[0][i + 1], stroke[1][i + 1]), color, self.thickness)
            if self.size != BASE_SIZE:
                img_batch[idx] = cv2.resize(img, (self.size, self.size))
            else:
                img_batch[idx] = img

            if TEST: # show bitmap example images
                cv2.imshow('result' + str(idx),img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        img_batch = (img_batch - 0) / 255 # normalization to 0 ~ 1
        img_batch = (img_batch - 0.5) / 0.5 # normalization to -1 ~ 1
        return torch.from_numpy(img_batch).float().unsqueeze(1)

def collate_fn(batch):
    # torch.stack changed to torch.cat
    if torch.is_tensor(batch[0]):
        out = None
        return torch.cat(batch, 0, out=out)
    elif isinstance(batch[0], dict):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    else:
        raise TypeError("Invalid batch type!")

def main():
    # test the code, Change the global var (in line 15) TEST = True for test
    # Remember to change TEST = False when use this model !!!
    # a example of how to use this model to load data
    stroke2img = Strokes2Imgs(size = 224)
    dataset = Strokes_Imgs_data(chunk_size = 2 , transform = stroke2img)
    dataloader = data.DataLoader(dataset, batch_size = 2,
                                    num_workers = 2, collate_fn = collate_fn)

    num_epoches = 2
    num_iter = 0
    for epoch in range(num_epoches):
        for d in dataloader:
            num_iter += 1
            #print(epoch, num_iter)
            print(d['drawing'])
            print(d['y'])
            if num_iter == 1:
                return 0

if __name__ == '__main__':
    main()
