import tensorflow as tf
import numpy as np
import pandas as pd
import sys

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeros elsewhere."""
    e = np.zeros((10))
    e[j] = 1.0
    return e

def read_dataset(input_file = 'kaggle_data/test.csv'):
	rawdata = pd.read_csv(input_file)
	X = rawdata[rawdata.columns[0:]].values
	return X,rawdata.shape[0]

X,N = read_dataset()

model_path = "model/basic.ckpt"
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, model_path)

"""

X,Y = read_dataset()
X,Y = shuffle(X,Y, random_state=1)

trainX,testX,trainY,testY = train_test_split(X,Y, test_size=0.2)

# define hyper-parameters
learning_rate = 0.05
total_epochs = 40
cost_history = np.empty(shape=[1], dtype = float)
in_dim = X.shape[1]
out_dim = Y.shape[1]
model_path = "model/basic.ckpt"

# define number of layers and neurons
n_hidden_1 = 30
n_hidden_2 = 20

x = tf.placeholder(tf.float32,[None, in_dim])
correct_y = tf.placeholder(tf.float32,[None, out_dim])

# define model
def NN(x, weights, biases):

	layer1 = tf.add(tf.matmul(x,weights['w1']),biases['b1'])
	layer1 = tf.nn.sigmoid(layer1)

	layer2 = tf.add(tf.matmul(layer1,weights['w2']),biases['b2'])
	layer2 = tf.nn.relu(layer2)

	out_layer = tf.matmul(layer2, weights['out']) + biases['out']
	return out_layer

# define weights and biases
weights = {
	'w1': tf.Variable(tf.truncated_normal([in_dim,n_hidden_1]),dtype=tf.float32),
	'w2': tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2]),dtype=tf.float32),
	'out': tf.Variable(tf.truncated_normal([n_hidden_2,out_dim]),dtype=tf.float32)
}

biases = {
	'b1': tf.Variable(tf.truncated_normal([n_hidden_1]),dtype=tf.float32),
	'b2': tf.Variable(tf.truncated_normal([n_hidden_2]),dtype=tf.float32),
	'out': tf.Variable(tf.truncated_normal([out_dim]),dtype=tf.float32)
}

# output of NN
y = NN(x, weights, biases)

# define cost function and optimizer
cost_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=correct_y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_fn)

save = tf.train.Saver()

sess = tf.Session()
if restore:
	save.restore(sess, model_path)
else:
	init = tf.global_variables_initializer()
	sess.run(init)
# fw = tf.summary.FileWriter('/home/anant/ml/mnist/graph',sess.graph)

mse_history = []
accuracy_history = []

for epoch in range(total_epochs):
	sess.run(training_step, feed_dict={x: trainX, correct_y: trainY})
	cost = sess.run(cost_fn, feed_dict={x: trainX, correct_y: trainY})
	cost_history = np.append(cost_history,cost)
	correct = tf.equal(tf.argmax(y,1), tf.argmax(correct_y,1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	print("Accuracy: ", (sess.run(accuracy, feed_dict={x: trainX, correct_y: trainY})))
	pred_y = sess.run(y,feed_dict={x: testX, correct_y: testY})
	mse = tf.reduce_mean(tf.square(pred_y - testY))
	_mse = sess.run(mse)
	mse_history.append(_mse)
	accuracy = (sess.run(accuracy, feed_dict={x: testX, correct_y: testY}))
	accuracy_history.append(accuracy)

	print("epoch:",epoch,", cost:",cost,", mse:",_mse, ", Train Accuracy:",accuracy)

save_path = save.save(sess,model_path)
print("Save in %s" % save_path)

# plt.plot(mse_history,'r')
# plt.show()
# plt.plot(accuracy_history)
# plt.show()

correct = tf.equal(tf.argmax(y,1), tf.argmax(correct_y,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print("Test Accuracy: ",(sess.run(accuracy, feed_dict={x: testX, correct_y: testY})))

sess.close()
"""