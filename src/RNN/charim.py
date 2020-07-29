
import tensorflow as tf
import  numpy as np
from tensorflow.contrib import rnn

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don&d assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

print(sentence)

char_set = list(set(sentence))
print(char_set)

char_dic = {w: i for i, w in enumerate(char_set)}
# print(char_dic)

hidden_size = 50
num_classes = len(char_set)
sequence_length = 10
learning_rate = 0.1

data_X = []
data_Y = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i: i + sequence_length] # h e l l
    y_str = sentence[i+1: i + sequence_length + 1] # e l l o
    print(x_str, ' -> ', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    data_X.append(x)
    data_Y.append(y)

batch = len(data_X)

# Network
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)

def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch, sequence_length, num_classes])

weights = tf.ones([batch, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights
)

mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X:data_X, Y:data_Y})

    for j, result in enumerate(results): # [sequence, num_classes]
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), 1)

result = sess.run(outputs, feed_dict={X:data_X})
for j, result in enumerate(result):
    index = np.argmax(result, axis=1)
    if j is 0:
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')
