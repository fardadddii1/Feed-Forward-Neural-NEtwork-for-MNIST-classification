# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:56:42 2021

@author: Briana Santo
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as mplp
import argparse
import warnings
import time
import os

cwd = os.getcwd()

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Parameters for training our FNN.')
parser.add_argument('-A','--learning_rate', metavar='', required = False, type=float, nargs='?', default=0.001,
                    help='determines the step size at each iteration while moving toward the minimum of the loss function; default = 0.001')
parser.add_argument('-B','--training_epochs', metavar='', required = False, type=int, nargs='?', default=20,
                    help='the number of complete passes through the training dataset; default = 20')
parser.add_argument('-C','--batch_size', metavar='', required = False, type=int, nargs='?', default=100,
                    help='the number of training samples processed before the model is updated; default = 100')
parser.add_argument('-D','--display_step', metavar='', required = False, type=int, nargs='?', default=1,
                    help='indicates how frequently the training loss and accuracy are reported; default = 1 [each epoch]')
parser.add_argument('-E','--num_hidden_1', metavar='', required = False, type=int, nargs='?', default=256,
                    help='the number of nodes in the first layer; default = 256')
parser.add_argument('-F','--num_hidden_2', metavar='', required = False, type=int, nargs='?', default=256,
                    help='the number of nodes in the second layer; default = 256')

args = parser.parse_args()


learning_rate = args.learning_rate
training_epochs = args.training_epochs
batch_size = args.batch_size
display_step = args.display_step

num_hidden_1 = args.num_hidden_1
num_hidden_2 = args.num_hidden_2

def reshape_data(data,im_size):
    data = data.reshape(im_size,im_size)
    return data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

train_data = mnist.train.images
test_data = mnist.test.images
val_data = mnist.validation.images

train_labels = mnist.train.labels
test_labels = mnist.test.labels
val_labels = mnist.validation.labels

num_train = mnist.train.num_examples
num_test = mnist.test.num_examples
num_val = mnist.validation.num_examples

num_classes = len(train_labels[-1])
num_features = mnist.train.images[-1].shape[0]

im_dims = mnist.train.images[-1].shape
im_min = mnist.train.images[-1].min()
im_max = mnist.train.images[-1].max()

#Preview training data
sample_train = mnist.train.images[-1]
sample_train = reshape_data(sample_train,im_size=28)
#mplp.imshow(sample_train)
#mplp.title('A MNIST Image')
#mplp.colorbar()
#mplp.show()
#time.sleep(3)
#mplp.close()


num_input = num_features
x = tf.placeholder('float', [None, num_input])
y = tf.placeholder('float', [None, num_classes])

w1 = tf.Variable(tf.random_normal([num_input, num_hidden_1]))
b1 = tf.Variable(tf.random_normal([num_hidden_1]))
layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

w2 = tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2]))
b2 = tf.Variable(tf.random_normal([num_hidden_2]))
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w2), b2))

output = tf.Variable(tf.random_normal([num_hidden_2, num_classes]))
b_out = tf.Variable(tf.random_normal([num_classes]))
layer_out = tf.matmul(layer_2, output) + b_out

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layer_out, labels = y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

y_preds = tf.argmax(layer_out,1)
y_trues = tf.argmax(y,1)
success = tf.equal(y_preds,y_trues)
accuracy = tf.reduce_mean(tf.cast(success,tf.float32))

cost_set = []
acc_set = []
epoch_set = []

cost_test_set = []
acc_test_set = []

init = tf.global_variables_initializer()

with tf.Session() as sess:
   sess.run(init)

   print('\n ---------- Initiating Training ---------- \n')

   for epoch in range(training_epochs):
      avg_cost = 0.
      avg_acc = 0.
      total_batch = int(num_train / batch_size)

      avg_test_cost = 0.
      avg_test_acc = 0.

      for i in range(total_batch):
         batch_xs, batch_ys = mnist.train.next_batch(batch_size)

         sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})

         avg_cost += sess.run(cost, feed_dict = {x: batch_xs, y: batch_ys}) / total_batch

         avg_acc += sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}) / total_batch

      if epoch % display_step == 0:
         print('Epoch:', '%04d' % (epoch + 1), 'Train Cost =', '{:.9f}'.format(avg_cost), 'Train Acc =', '{:.3f}'.format(avg_acc))
      cost_set.append(avg_cost)
      acc_set.append(avg_acc)
      epoch_set.append(epoch + 1)

      avg_test_cost += sess.run(cost, feed_dict = {x: test_data, y: test_labels})

      avg_test_acc += sess.run(accuracy, feed_dict={x:test_data, y:test_labels})

      cost_test_set.append(avg_test_cost)
      acc_test_set.append(avg_test_acc)

   print('\n ---------- Training Complete ---------- \n')

   fig = mplp.plot(epoch_set, cost_set, 'o', label = 'Training set')
   mplp.plot(epoch_set, cost_test_set, 'o', label = 'Testing set')
   mplp.ylabel('cost')
   mplp.xlabel('epoch')
   mplp.title('MNIST Classification Loss')
   mplp.legend()
   fname = cwd + '/' + 'loss.png'
   mplp.savefig(fname)
   mplp.close()
   #mplp.show()

   fig = mplp.plot(epoch_set, acc_set, 'o', label = 'Training set')
   mplp.plot(epoch_set, acc_test_set, 'o', label = 'Testing set')
   mplp.ylabel('cost')
   mplp.xlabel('epoch')
   mplp.title('MNIST Classification Accuracy')
   mplp.legend()
   fname = cwd + '/' + 'acc.png'
   mplp.savefig(fname)
   mplp.close()
   #mplp.show()

   print('Final Model Accuracy on Test Set:', accuracy.eval({x: test_data, y: test_labels}))
   print('\n')
