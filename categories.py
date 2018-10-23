import os

file_list = os.listdir('./data/train_simplified_strokes')
categories = [f.split('.')[0] for f in file_list]
categories = sorted(categories, key=str.lower)
with open('categories.txt','w') as f:
    for idx, cat in enumerate(categories):
        f.write(str(idx) + ' ' + cat.replace(' ', '_') + '\n')
