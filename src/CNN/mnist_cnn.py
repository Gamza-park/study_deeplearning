import tensorflow as tf
import time, os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# model define
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10]) # 0 0 0 0 1 0 0 0 0 0

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))  # [W, H, c_in, num_filters (c_out)]
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # [?, 28, 28, 32]
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # [?, 14, 14, 32]

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))  # [W, H, c_in, num_filters (c_out)]
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')  # [?, 14, 14, 64]
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # [?, 7, 7, 64]

L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])

W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.nn.xw_plus_b(L2_flat, W3, b, name='hypothesis')

# W4 = tf.Variable(tf.random_normal([200, 10]))
# b4 = tf.Variable(tf.random_normal([10]))
# hypothesis = tf.matmul(L3, W4) + b4 # 0~9

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # True -> 1.0 False -> 0.0

# define cost/loss & optimizer

# hypothesis = tf.nn.softmax(hypothesis)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y)) # (batch, 1) -> (1, ) (dimension)

learning_rate = 0.001
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())


summary_op = tf.summary.scalar("accuracy", accuracy)

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
val_summary_dir = os.path.join(out_dir, "summaries", "dev")
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)


training_epochs = 5
batch_size = 100

# train my model
max = 0.0
early_stopped = 0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size) # 55000/100 = 550
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _, a = sess.run([cost, optimizer, summary_op], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'training cost =', '{:.9f}'.format(avg_cost))
    train_summary_writer.add_summary(a, early_stopped)

    test_accuracy, summaries = sess.run([accuracy, summary_op], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    val_summary_writer.add_summary(summaries, early_stopped)
    print('Test Accuracy:', test_accuracy)
    if test_accuracy > max:
        max = test_accuracy
        early_stopped = epoch + 1
        saver.save(sess, checkpoint_prefix, global_step=early_stopped)

print('Learning Finished!')
print('Test Max Accuracy:', max)
print('Early stopped time:', early_stopped)

