# This is a program that uses the TensorFlow library to train a model on the MNIST Dataset.

# The learning algorithm used is Softmax Regression.

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Setting up feature matrix
x = tf.placeholder(tf.float32, [None, 784])

# Initialising weights matrix
W = tf.Variable(tf.zeros([784, 10]))

# Setting up Bias vector for regularisation
bias = tf.Variable(tf.zeros([10]))

# Producing hypothesis
hypothesis = tf.nn.softmax(tf.matmul(x, W) + bias)

# Loading target values
targets = tf.placeholder(tf.float32, [None, 10])

# Framework for calculating cost using softmax activation function
cost_value = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=hypothesis)

# Training step
train = tf.train.GradientDescentOptimizer(0.05).minimize(cost_value)

# Running the session

sess = tf.InteractiveSession()
# Initialise all variables
tf.global_variables_initializer().run()

# Train in batches of 100

for i in range(12000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train, feed_dict={x:batch_x, targets: batch_y})

# Testing
correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

print "Accuracy of prediction of digits is: %s" %(sess.run(accuracy*100, feed_dict={x: mnist.test.images, targets: mnist.test.labels}))













