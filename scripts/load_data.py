import os
import json
import numpy as np

# "above": 10,
# "below": -1,
# "bottom": 2394,
# "height": 27,
# "is_table": true,
# "left": 291,
# "next": 991,
# "prev": 13,
# "right": 449,
# "top": 2367,
# "width": 158,
# "word": "references"


def loadJson(path):
    print(path)
    with open(path) as data_file:
        data = json.load(data_file)
        num_word = len(data)
        x = np.zeros((num_word, 10))
        y = np.zeros((num_word, 2))
        id = []
        for i in range(len(data)):
            data_one = data[i]
            left = data_one['left']
            top = data_one['top']
            right = data_one['right']
            bottom = data_one['bottom']
            above = data_one['above']
            below = data_one['below']
            prev = data_one['prev']
            next = data_one['next']
            width = data_one['width']
            height = data_one['height']
            is_table = data_one['is_table']
            idd = data_one['image_id']
            xx = np.zeros(10);
            xx[0] = left
            xx[1] = top
            xx[2] = right
            xx[3] = bottom
            xx[4] = above
            xx[5] = below
            xx[6] = prev
            xx[7] = next
            xx[8] = width
            xx[9] = height
            x[i] = xx
            yy = np.zeros(2)
            yy[int(is_table)] = 1
            y[i] = yy
            id.append(idd)
            # y[i] = yy
        return x, y, id


def load_data(dir):
    init = False
    ID = []
    for i in os.listdir(dir):
        if not i.endswith('json'):
            continue
        x, y, id = loadJson(dir + '/' + i)
        if not init:
            X = x
            Y = y
            init = True
        else:
            X = np.append(X, x, axis=0)
            Y = np.append(Y, y, axis=0)
            ID = ID + id
    print(np.shape(X), np.shape(Y))
    return X, Y, ID