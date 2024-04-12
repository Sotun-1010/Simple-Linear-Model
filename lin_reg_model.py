import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# for W = 3 and b = 2 [5.  8. 11. 14.]
# for W = 1 and b = -2 [-1.  0.  1.  2]
# for W = 0 and b = -1 [-1. -1. -1. -1.]
# for W = 1 and b = -1 [0. 1. 2. 3.]
# for W = -1 and b = 1 [0. -1. -2. -3.]

W = tf.Variable([1], dtype = tf.float32)
b = tf.Variable([-1], dtype = tf.float32)

x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

Linear_Model = W * x + b
loss  = tf.reduce_sum(tf.square(Linear_Model-y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

session = tf.Session()

init = tf.global_variables_initializer()
session.run(init)
print(session.run(loss, {x: x_train, y: y_train}))

# print(session.run(linear_model, {x: [x_train]}))
 
for i in range(1000):
    session.run(train, {x:x_train, y:y_train})

new_W, new_b, new_loss = session.run([W, b, loss], {x:x_train, y:y_train})

print('New_W is %s'%new_W)
print('New_b is %s'%new_b)
print('New_loss is %s'%new_loss)

print(session.run(Linear_Model, {x: [10, 20, 30, 40, 50]}))