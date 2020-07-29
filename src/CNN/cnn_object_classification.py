import tensorflow as tf
import time, os
import numpy as np

from tensorflow.keras.datasets.cifar10 import load_data

# model 정의
X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y') # 0 0 0 0 1 0 0 0 0 0
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Layer 1
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# Layer 2
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# Layer 3
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)


L3_flat = tf.reshape(L3, [-1, 4 * 4 * 128])

W4 = tf.get_variable("W4", shape=[4 * 4 * 128, 128], initializer=tf.initializers.he_normal())
b4 = tf.Variable(tf.random_normal([128]))
FC1 = tf.nn.relu(tf.nn.xw_plus_b(L3_flat, W4, b4))
FC1 = tf.nn.dropout(FC1, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[128,64], initializer=tf.initializers.he_normal())
b5= tf.Variable(tf.random_normal([64]))
FC2 = tf.nn.relu(tf.nn.xw_plus_b(FC1, W5, b5))
FC2 = tf.nn.dropout(FC2, keep_prob=keep_prob)

W6 = tf.get_variable("W6", shape=[64, 10], initializer=tf.initializers.he_normal())
b6 = tf.Variable(tf.random_normal([10]))

hypothesis = tf.nn.xw_plus_b(FC2, W6, b6, name='hypothesis')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # True -> 1.0 False -> 0.0

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y)) # (batch, 1) -> (1, ) (dimension)

learning_rate = 0.001

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

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

(x_train_val, y_train_val),(x_test, y_test) = load_data()

# 1. Shuffling
shuffle_indices = np.random.permutation(np.arange(len(y_train_val)))
shuffled_x = np.asarray(x_train_val[shuffle_indices])
shuffled_y = y_train_val[shuffle_indices]

# 2. Validation
dev_sample_index = -1 * int(0.1 * float(len(y_train_val)))
x_train, x_val = shuffled_x[:dev_sample_index], shuffled_x[dev_sample_index:]
y_train, y_val = shuffled_y[:dev_sample_index], shuffled_y[dev_sample_index:]

# 3. Y label -> ont-hot
y_train_one_hot = np.eye(10)[y_train] # (10, 10, 1)
y_train_one_hot = np.squeeze(y_train_one_hot, axis=1)

y_test_one_hot = np.eye(10)[y_test] # (10, 10, 1)
y_test_one_hot = np.squeeze(y_test_one_hot, axis=1)

y_val_one_hot = np.eye(10)[y_val] # (10, 10, 1)
y_val_one_hot = np.squeeze(y_val_one_hot, axis=1)

def next_batch(batch_size, data):
    data = np.array(data)
    np.random.seed(10)

    shuffle_indices = np.random.permutation(np.arange(len(data)))
    shuffled_data = data[shuffle_indices]
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1)*batch_size, len(data))
        yield shuffled_data[start_index:end_index]



for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(y_train) / batch_size) # 55000/100 = 550

    batches = next_batch(batch_size, list(zip(x_train, y_train_one_hot)))
    for batch in batches:
        batch_xs, batch_ys = zip(*batch)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.8}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'training cost =', '{:.9f}'.format(avg_cost))
    test_accuracy = sess.run(accuracy, feed_dict={X: x_val, Y:y_val_one_hot , keep_prob: 1.0})
    print('Test Accuracy:', test_accuracy)
    if test_accuracy > max:
        max = test_accuracy
        early_stopped = epoch + 1
        saver.save(sess, checkpoint_prefix, global_step=early_stopped)

print('Learning Finished!')
print('Test Max Accuracy:', max)
print('Early stopped time:', early_stopped)


