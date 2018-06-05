import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

print("1 = ", node1)
print("2 = ", node2)
print("3 = ", node3)

sess = tf.Session()

print("sess 1&2 = ", sess.run([node1,node2]))
print("sess 3 = ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_sum = a+b

print("add_sum = ", sess.run(add_sum, feed_dict={a:1,b:3}))

