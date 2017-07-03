from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


tf.reset_default_graph()
# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 100
display_step = training_epochs / 10

# Network Parameters
n_hidden_1 = 20  # 1st layer number of features
n_hidden_2 = 5  # 2nd layer number of features
n_input = 10  # MNIST data input (img shape: 28*28)
n_classes = 2  # MNIST total classes (0-9 digits

# Create model
def multilayer_perceptron(x, weights, biases, kp):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #     layer_1 = tf.nn.dropout(layer_1, kp)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #     layer_2 = tf.nn.dropout(layer_2, kp)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def train_and_get_accuracy(X,Y,X_t,Y_t,X_v, Y_v):
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    xv = tf.placeholder("float", [None, n_input])
    yv = tf.placeholder("float", [None, n_classes])

    kp = tf.placeholder(tf.float32)
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))

    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases, kp)
    validation_pred = multilayer_perceptron(xv, weights, biases, kp)
    validation_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=validation_pred, labels=yv))

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cost = cost + 0*sum(reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    tf.summary.scalar("training-loss", cost)
    tf.summary.scalar("validation-loss", validation_cost)

    summary_op = tf.summary.merge_all()

    logs_path = 'logs'

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        batch_x = X
        batch_y = Y

        batch_xv = X_v
        batch_yv = Y_v
        i = 0
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            avg_cost_validation = 0.
            total_batch = 1
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, cv, summary = sess.run([optimizer, cost, validation_cost, summary_op], feed_dict={x: batch_x,
                                                          y: batch_y, xv: batch_xv, yv: batch_yv, kp: 0.5})
            # Compute average loss
            avg_cost += c / total_batch
            avg_cost_validation += cv / total_batch
            writer.add_summary(summary, i)
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost), "validation-cost=", \
                    "{:.9f}".format(avg_cost_validation))
            i += 1
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        acc=accuracy.eval({x: X_t, y: Y_t, kp: 1});
        print("Accuracy:", acc)
        return tf.argmax(pred, 1).eval({x: X_t, y: Y_t, kp: 1})
