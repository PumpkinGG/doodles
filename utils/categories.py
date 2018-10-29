import os
import pandas as pd

def write_category():
    file_list = os.listdir('../data/train_simplified_strokes')
    categories = [f.split('.')[0] for f in file_list]
    categories = sorted(categories, key=str.lower)
    with open('categories.txt','w') as f:
        for idx, cat in enumerate(categories):
            f.write(str(idx) + ' ' + cat.replace(' ', '_') + '\n')


############################################################################
def read_category(file = './categories.txt'):
    if not os.path.exists(file):
        raise FileNotFoundError('No such file.')

    category = {}

    with open(file, 'r') as f:
        for line in f.readlines():
            idx = int(line.strip().split(' ')[0])
            category[idx] = line.strip().split(' ')[1]

    return category


############################################################################
def cal_example_num():
    path = './data/train'
    files = os.listdir(path)
    min_num = 1e10
    total_len = 0
    for f in files:
        df = pd.read_csv(os.path.join(path, f), compression = 'gzip', iterator = False)
        length = len(df)
        total_len += length
        if length < min_num:
            min_num = length
    print('Min num of train example in one file is %d, \n Total train example num is: %d' % (min_num, total_len))

    min_num = 1e10
    path = './data/val'
    files = os.listdir(path)
    total_len = 0
    for f in files:
        df = pd.read_csv(os.path.join(path, f), compression = 'gzip', iterator = False)
        length = len(df)
        total_len += length
        if length < min_num:
            min_num = length
    print('Min num of train example in one file is %d, \n Total train example num is: %d' % (min_num, total_len))

if __name__ == '__main__':
    cal_example_num()
    # write_category()
# Min num of train example in one file is 101261,
#  Total train example num is: 8160122
# Min num of train example in one file is 101550,
#  Total train example num is: 2039878
