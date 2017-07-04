import load_data
import network_perceptron
import numpy as np
import json

train_dir = '/home/srq/Playground/unlv-train/train'
validation_dir = '/home/srq/Playground/unlv-train/train/validation'
test_dir = '/home/srq/Playground/unlv-train/test'
test_out_dir = '/home/srq/Playground/unlv-train/test/outs'

X, Y,temp = load_data.load_data(train_dir)
X_t, Y_t,ID_t = load_data.load_data(test_dir)
X_v, Y_v,temp = load_data.load_data(validation_dir)

predictions = network_perceptron.train_and_get_accuracy(X, Y, X_t, Y_t, X_v, Y_v)

print(predictions)
print(np.shape(predictions))
print(np.shape(X_t))



init = False
last_image_id = ''
last_image = []

ii = 0
for i in ID_t:
    left = X_t[ii][0]
    top = X_t[ii][1]
    right = X_t[ii][2]
    bottom = X_t[ii][3]
    is_table = predictions[ii]
    word = {}
    word['left'] = float(left)
    word['top'] = float(top)
    word['right'] = float(right)
    word['bottom'] = float(bottom)
    word['is_table'] = float(is_table)

    if not init:
        last_image.append(word)
        init = True
    elif i == last_image_id:
        last_image.append(word)
    else:
        with open(test_out_dir + '/' + last_image_id + '.json', 'w') as outfile:
            json.dump(last_image, outfile)
        last_image = []
        last_image.append(word)

    last_image_id = i
    ii += 1

# print(last_image)

if len(last_image) != 0:
    with open(test_out_dir + '/' + last_image_id + '.json', 'w') as outfile:
        json.dump(last_image, outfile)
