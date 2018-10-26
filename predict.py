import os
import ast
import datetime as dt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from train import *
from load_data import *

class TestDataset(data.Dataset):
    def __init__(self, csv_dir, chunk_size = 1, transform = None):
        self.raw_strokes = pd.read_csv(csv_dir, iterator = True, usecols = ['key_id', 'drawing'])
        self.transform = transform
        self.chunk_size = chunk_size

    def __len__(self):
        return 112199 // self.chunk_size

    def __getitem__(self, idx):
        temp = self.raw_strokes.get_chunk(self.chunk_size)
        drawing = np.array(temp['drawing'].apply(ast.literal_eval))
        key_id = np.array(temp['key_id'])

        sample = {'key_id': torch.from_numpy(key_id),
                  'drawing': drawing}

        if self.transform:
            sample['drawing'] = self.transform(sample)

        return sample

def read_category(file = './categories.txt'):
    if not os.path.exists(file):
        raise FileNotFoundError('No such file.')

    category = {}

    with open(file, 'r') as f:
        for line in f.readlines():
            idx = int(line.strip().split(' ')[0])
            category[idx] = line.strip().split(' ')[1]

    return category

def predict(csv_dir = './data/test_simplified.csv', output_dir = './data/submision.csv'):
    start = dt.datetime.now()

    dtype = torch.float
    device = torch.device('cuda:0')
    # load predict model
    model, _ = load_model_optim(is_training = False)
    category = read_category(file = './categories.txt')
    # load model to GPU
    model.to(device)

    stroke2img = Strokes2Imgs(size = 224)

    dataset = TestDataset(csv_dir = csv_dir, chunk_size = 1, transform = stroke2img)
    # DataLoader use more than one num_workers will cause error (list out of range in load_data.Strokes2Imgs)
    dataloader = data.DataLoader(dataset, batch_size = 16,
                                    num_workers = 1, collate_fn = collate_fn)
    model.eval()
    iter = 0
    for d in dataloader:
        inputs = d['drawing'].to(device)
        key_id = d['key_id']

        with torch.no_grad():
            outputs = model(inputs)

        # calculate acc_in
        _, index = torch.sort(outputs, dim = 1, descending=True)
        pred = index[:, 0:3].tolist() # take top-3 scores predict

        word = []
        for p in pred:
            word.append(category[p[0]] + ' ' + category[p[1]] + ' ' + category[p[2]])

        chunk = pd.DataFrame({'key_id': key_id, 'word': word})
        if iter == 0:
            chunk.to_csv(output_dir, index=False)
        else:
            chunk.to_csv(output_dir, mode='a', header=False, index=False)

        if iter % 100 == 99:
            print('%d iterations done.' % (iter + 1))

        iter += 1

    end = dt.datetime.now()
    print('Over, cost {}s'.format((end - start).seconds))

def main():
    predict(csv_dir = './data/test_simplified.csv',
            output_dir = './data/submision.csv')

if __name__ == '__main__':
    main()
