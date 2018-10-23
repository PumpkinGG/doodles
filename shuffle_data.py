import pandas as pd
import numpy as np
import datetime as dt
import os
import ast
from tqdm import tqdm

class Simplified_data(object):
    def __init__(self, input_path = './data'):
        self.input_path = input_path

    def f2c(self, filename):
        # split classname from file name
        assert isinstance(filename, str)
        return filename.split('.')[0]

    def list_categories(self):
        # return sorted categories
        files = os.listdir(os.path.join(self.input_path,
                            'train_simplified_strokes'))
        return sorted([self.f2c(f) for f in files], key=str.lower)

    def read_training_csv(self, category,
                            nrows=None, usecols=None,
                            drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified_strokes', category + '.csv'),
                        nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(ast.literal_eval)
            # ast.literal_eval("[[1, 2], [3, 4]]") = [[1, 2], [3, 4]] (str-->list)
        return df

def parse_datas(num_files=100, out_dir='./data/parsed_train_data'):
    start = dt.datetime.now()

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    s_data = Simplified_data('./data')
    categories = s_data.list_categories()
    print(len(categories))
    print(categories[:5])

    for y, c in tqdm(enumerate(categories)):
        df = s_data.read_training_csv(c, nrows=25000)
        df['y'] = y
        df['cv'] = (df.key_id // 10 ** 7) % num_files
        for k in range(num_files):
            filename = 'train_k{}.csv'.format(k)
            filename = os.path.join(out_dir, filename)
            chunk = df[df.cv == k]
            chunk = chunk.drop(['key_id'], axis=1) # drop key_id colume
            if y == 0:
                chunk.to_csv(filename, index=False)
            else:
                chunk.to_csv(filename, mode='a', header=False, index=False)

    for k in tqdm(range(num_files)):
        filename = 'train_k{}.csv'.format(k)
        filename = os.path.join(out_dir, filename)
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['rnd'] = np.random.rand(len(df))
            df = df.sort_values(by='rnd').drop('rnd', axis=1)
            df.to_csv(filename + '.gz', compression='gzip', index=False)
            os.remove(filename)
    print(df.shape)

    end = dt.datetime.now()
    print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))

def main():
    parse_datas(num_files=100)

if __name__ == '__main__':
    main()
