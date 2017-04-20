import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle

# TODO: Load traffic signs data.
training_file = "train.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X, y = train['features'], train['labels']

n_examples = X.shape[0]
image_shape = X.shape[1:]
n_classes = len(set(y))

print("Number of total training examples =", n_examples)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.02, train_size=0.2, random_state=42)

n_train = X_train.shape[0]
n_valid = X_valid.shape[0]
print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, [227, 227])
y = tf.placeholder(tf.int32, (None))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], n_classes)  # use this shape for the weight matrix
# fc8, 1000
mu = 0
sigma = 0.1
fc8W = tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))
fc8b = tf.Variable(tf.zeros(n_classes))
logits = tf.matmul(fc7, fc8W) + fc8b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001
EPOCHS = 5
BATCH_SIZE = 128

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# TODO: Train and evaluate the feature extraction model.

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

import sys

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            print("Training batch {} of {} at epoch {}".format(int(offset/BATCH_SIZE), int(num_examples/BATCH_SIZE), i+1))
            sys.stdout.flush()
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        sys.stdout.flush()
        
    saver.save(sess, './alexnet_traffic_sign')
    print("Model saved")
