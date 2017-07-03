import load_data
import network_perceptron

train_dir = '/home/srq/Playground/unlv-train/train'
validation_dir = '/home/srq/Playground/unlv-train/train/validation'
test_dir = '/home/srq/Playground/unlv-train/test'

X, Y = load_data.load_data(train_dir)
X_t, Y_t = load_data.load_data(test_dir)
X_v, Y_v = load_data.load_data(validation_dir)

network_perceptron.train_and_get_accuracy(X, Y, X_t, Y_t, X_v, Y_v)