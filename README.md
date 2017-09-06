## Tesorflow
"**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  The graph nodes represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow also includes TensorBoard, a data visualization toolkit."       
-- words from tensorflow documentation
## Installation
*See [Installing TensorFlow](https://www.tensorflow.org/get_started/os_setup.html) for instructions on how to install our release binaries or how to build from source.*

#### *Try your first TensorFlow program*
```shell
$ python
```
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
'Hello, TensorFlow!'
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a + b)
42
>>> sess.close()
```
#### see the file in action [gettingstarted.py](https://github.com/kakshay21/ML-tensorflow/blob/master/gettingstarted.py)

## Let's begin with MNIST
I would recommend to follow [this guide](https://www.tensorflow.org/get_started/mnist/beginners)

The guide is for begineers and really helpful to follow next.

Now Let's see the 5 layered neural network [here](https://github.com/kakshay21/ML-tensorflow/blob/master/mnistv2.py)

Now on increasing one more hidden layer, [check this out](https://github.com/kakshay21/ML-tensorflow/blob/master/mnistv3.py)
