import numpy as np
import time, os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNiST_data/", one_hot=True, validation_size=5000)

# print(np.shape(mnist.validation.images))
# print(np.shape(mnist.validation.labels)) # [0 0 0 0 1 0 0 0 0 ]
# print(np.shape(mnist.train.images))
# print(np.shape(mnist.train.labels)) # [0 0 0 0 1 0 0 0 0 ]
# print(np.shape(mnist.test.images))
# print(np.shape(mnist.test.labels)) # [0 0 0 0 1 0 0 0 0 ]

X = tf.placeholder(tf.float32, [None, 784], name="X") # [None , 784]
Y = tf.placeholder(tf.float32, [None, 10], name="Y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
l2_loss = 0

W1 = tf.Variable(tf.random_normal([784,256]))
# W1 = tf.get_variable("W2", shape=[784,256],initializer=tf.contrib.layers.xavier_initializer()) # xavier
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1) # result [None , 256]
# L1 = tf.nn.dropout(L1,keep_prob=keep_prob) # drob out
l2_loss += tf.nn.l2_loss(W1)
l2_loss += tf.nn.l2_loss(b1)

W2 = tf.Variable(tf.random_normal([256,300]))
b2 = tf.Variable(tf.random_normal([300]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([300,10]))
b3 = tf.Variable(tf.random_normal([10]))# [None, 10]
# hypothesis = tf.matmul(L2,W3) +b3
hypothesis = tf.nn.xw_plus_b(L2,W3,b3, name="hypothesis") #[0.2 0.1 0.7][0 0 1]

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y)) + l2_loss *0.001

# model saving
summary_op = tf.summary.scalar("accuracy", accuracy)
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries","train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph)
checkpoint_dir = os.path.abspath(os.path.join(out_dir,"checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "mode1")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3) # 최근으로부터 3개

# optimization
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# epoch, batch_size
training_epochs = 10
batch_size = 100

max = 0.0
early_stopped = 0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.5}
        c, _, a = sess.run([cost, optimizer, summary_op], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch {}, training cost {}'.format(epoch,avg_cost))
    train_summary_writer.add_summary(a,early_stopped)

    feed_dict = {X: mnist.validation.images, Y: mnist.validation.labels, keep_prob: 1.0}
    valid_accuracy = sess.run(accuracy, feed_dict=feed_dict)
    print('Validation accuracy', valid_accuracy)

    if max < valid_accuracy:
        max = valid_accuracy
        early_stopped = epoch + 1
        saver.save(sess, checkpoint_prefix, global_step=early_stopped)

print('Learning finished!')
print('Validation Accuracy', max)
print('Early stopped time', early_stopped)

feed_dict = {X:mnist.test.images, Y:mnist.test.labels, keep_prob: 1.0}
test_accuracy = sess.run(accuracy, feed_dict=feed_dict)
print('Test accuracy', test_accuracy)
