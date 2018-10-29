import torch
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd
import datetime as dt

import nets
import utils

PREDICT_MODEL = './checkpoint/cnn_pretrained.pth'
SUB_FILE_DIR = './data/submision.csv'

def test_dataset(data.Dataset):
    def __init__(self, file_dir = './data/test_simplified.csv', transform = None):
        self.data_file = pd.read_csv(file_dir)

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        temp = self.data_file.iloc[idx, :]

        drawing = np.array(ast.literal_eval(temp['drawing']))

        sample = {'drawing': drawing,
                   'y': [],
                   'cache': temp.drop(['countrycode', 'drawing'], axis = 1)}

        if self.transform:
            sample = self.transform(sample)

        return sample

def run_test():
    # define using gpu or cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    # define network
    model = nets.MobileNetV2(utils.NUM_CLASSES)

    checkpoint = torch.load(PREDICT_MODEL)
    model.load_state_dict(checkpoint['model_state_dict'])

    # load model params to cuda/GPU
    model = model.to(device)

    start = dt.datetime.now()
    print('# testing......................')
    transform = utils.Points2Imgs(size = 224)
    dataset = test_dataset(transform = transform)
    dataloader = data.DataLoader(dataset, batch_size = 32,
                                    num_workers = 8, collate_fn = utils.null_imgs_collate)

    category = utils.read_category('./utils/categories.txt')

    model.eval()
    i = 0

    for input, _, cache in dataloader:

        input = input.to(device)

        with torch.no_grad():
            logit = model(input)
            prob = F.softmax(logit, 1)
            value, top = prob.topk(3, dim=1, largest=True, sorted=True)

        predict = []
        for line in top:
            predict.append(category[line[0]] + ' ' + category[line[1]] + ' ' + category[line[2]])

        cache['word'] = predict
        if i == 0:
            cache.to_csv(SUB_FILE_DIR, index=False)
        else:
            cache.to_csv(SUB_FILE_DIR, mode='a', header=False, index=False)

        if i%500 == 499:
            print('%6d Iteration finished.' % i + 1)

        i += 1

    end = dt.datetime.now()
    print('Finished all. Cost {}'.fomat(end - start))
