import tensorflow as tf
''' Basic idea
input > weight > hidden layer 1 (activation function)>weights
>hidden layer 2 (activation function) > weights > output layer
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("A/", one_hot=True)

n_nodes_hl1=500
n_nodes_hl2=500
n_nodes_hl3=500
n_classes=10
batch_size=100
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def neuralNetworkModel(data):
	hidden_1_Layer={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_Layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_Layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_Layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
	l1=tf.add(tf.matmul(data, hidden_1_Layer['weights']),hidden_1_Layer['biases'])
	l1=tf.nn.relu(l1)
	l2=tf.add(tf.matmul(l1, hidden_2_Layer['weights']),hidden_2_Layer['biases'])
	l2=tf.nn.relu(l2)
	l3=tf.add(tf.matmul(l2, hidden_3_Layer['weights']),hidden_3_Layer['biases'])
	l3=tf.nn.relu(l3)
	output=tf.matmul(l3, output_Layer['weights'])+output_Layer['biases']
	return output
def trainNeuralNetwork(x):
	prediction=neuralNetworkModel(x)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer=tf.train.AdamOptimizer().minimize(cost)
	hm_epochs=10
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(hm_epochs):
			epoch_loss=0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y=mnist.train.next_batch(batch_size)
				_, c=sess.run([optimizer,cost],feed_dict={x: epoch_x,y: epoch_y})
				epoch_loss+=c
			print('Epoch',epoch,'completed out of ',hm_epochs,'loss: ',epoch_loss)
		correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy: ',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

trainNeuralNetwork(x)
