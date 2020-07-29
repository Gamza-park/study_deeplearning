import tensorflow as tf

# x_data = [[1,2],[3,4],[5,6]]
x_data = [[1,3]]

# shape = [2] -> x_data = [1,2]

x = tf.placeholder(tf.float32, shape = [None,2])

w = tf.Variable(tf.random_normal([2,1], name = 'weight')) # 학습이 되는 변수 # 랜덤하게 노말한정수값
b = tf.Variable(tf.random_normal([1],name = 'bias')) # 1차원배열이 나오기때문에 7번줄과 9번줄 참고
# random_uniform = 정말 랜덤
# random_normal  = 정규분포 랜덤

output = tf.sigmoid(tf.matmul(x, w) + b) # activation function

# Session
# sess = tf.Session() # Session 실행방법 1
with tf.Session() as sess:
    # 랜덤 변수 초기화
    sess.run(tf.global_variables_initializer())
    prediction = sess.run(output, feed_dict={x: x_data})
    print(prediction)
#with이외의 구역에서는 Session활성화 x