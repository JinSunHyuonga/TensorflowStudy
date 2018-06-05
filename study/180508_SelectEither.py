import tensorflow as tf
import pandas as pd

input_data = pd.read_csv("./bear_data.csv", usecols=[0, 1])
output_data = pd.read_csv("./bear_data.csv", usecols=[2, 3])

Num_features = 2

Num_layer1 = 4
Num_layer2 = 5
Num_layer3  = 4

Classes = 2

Learning_rate = 1.2

x = tf.placeholder(tf.float32, [None, Num_features])
y_ = tf.placeholder(tf.float32, [None, Classes])

Weight_1 = tf.Variable(tf.zeros(shape=[Num_features, Num_layer1]), dtype=tf.float32, name = 'Weight_1')
Bias_1 = tf.Variable(tf.zeros(shape=[Num_layer1]), dtype=tf.float32, name = 'Bias_1')

Weight_2 = tf.Variable(tf.zeros(shape=[Num_layer1, Num_layer2]), dtype=tf.float32, name = 'Weight_2')
Bias_2 = tf.Variable(tf.zeros(shape=[Num_layer2]), dtype=tf.float32, name = 'Bias_2')

Weight_3 = tf.Variable(tf.zeros(shape=[Num_layer2, Num_layer3]), dtype=tf.float32, name = 'Weight_3')
Bias_3 = tf.Variable(tf.zeros(shape=[Num_layer3]), dtype=tf.float32, name = 'Bias_3')

Weight_out = tf.Variable(tf.zeros(shape=[Num_layer3, Classes]), dtype=tf.float32, name = 'Weight_out')
Bias_out = tf.Variable(tf.zeros(shape=[Classes]), dtype=tf.float32, name = 'Bias_out')

param_list = [Weight_1, Weight_2, Weight_3, Weight_out, Bias_1, Bias_2, Bias_3, Bias_out]
saver = tf.train.Saver(param_list)

layer_1 = tf.sigmoid(tf.matmul(x, Weight_1) + Bias_1)
layer_2 = tf.sigmoid(tf.matmul(layer_1, Weight_2) + Bias_2)
layer_3 = tf.sigmoid(tf.matmul(layer_2, Weight_3) + Bias_3)
out = tf.sigmoid(tf.matmul(layer_3, Weight_out) + Bias_out)

y = tf.nn.softmax(out)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(Learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(20000):
    _, cost = sess.run([train_step, cross_entropy], feed_dict = {x: input_data, y_: output_data})
    if i % 1000 == 0:
        print ("step : ", i)
        print ("cost : ", cost)
        print ("--------------")

correct_prediction = tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_,1)), tf.float32)

accuracy = tf.reduce_mean(correct_prediction)
sess.run(accuracy, feed_dict = {x: input_data, y_: output_data})

