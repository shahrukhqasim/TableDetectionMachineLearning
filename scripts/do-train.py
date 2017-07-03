import load_data
import network_perceptron
import numpy as np

train_dir = '/home/srq/Playground/unlv-train/train'
validation_dir = '/home/srq/Playground/unlv-train/train/validation'
test_dir = '/home/srq/Playground/unlv-train/test'

X, Y,temp = load_data.load_data(train_dir)
X_t, Y_t,ID_t = load_data.load_data(test_dir)
X_v, Y_v,temp = load_data.load_data(validation_dir)

predictions = network_perceptron.train_and_get_accuracy(X, Y, X_t, Y_t, X_v, Y_v)

print(predictions)
print(np.shape(predictions))
print(np.shape(X_t))
