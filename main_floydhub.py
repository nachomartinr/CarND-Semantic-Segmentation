import os.path
import tensorflow as tf
import helper
import warnings
import time
from datetime import timedelta
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np

STDDEV_INI = 0.01 # Standard deviation for default weight initialization
L2_REG = 0.001     # L2 regularization scaler
KEEP_PROB = 0.6
LEARNING_RATE = 0.0009

EPOCHS = 50
BATCH_SIZE = 16

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # load model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    with tf.name_scope('VGG16'):
        # extract variables
        graph = tf.get_default_graph()

        image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

        # Create summary to visualize layers
        #tf.summary.image('image_input', image_input)
        #tf.summary.histogram('layer3_out', layer3_out)
        #tf.summary.histogram('layer4_out', layer4_out)
        #tf.summary.histogram('layer7_out', layer7_out)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    with tf.name_scope('FCN'):
    	# According to this:
        # https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100
        # scaling the skip layers helps reducing the loss
        layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='layer3_out_scaled')
        layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='layer4_out_scaled')

        # 1x1 convolution of vgg layer 7
        fcn_layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes,
                                          kernel_size=1, strides=(1,1), padding='SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV_INI),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
                                          name='fcn_layer7_1x1')
        # upsample
        fcn_layer7_up = tf.layers.conv2d_transpose(fcn_layer7_1x1, num_classes,
                                                   kernel_size=4, strides=(2,2), padding='SAME',
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV_INI), 
                                                   name='fcn_layer7_up')

        # 1x1 convolution of vgg layer 4
        fcn_layer4_1x1 = tf.layers.conv2d(layer4_out_scaled, num_classes,
                                          kernel_size=1, strides=(1,1), padding='SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV_INI),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
                                          name='fcn_layer4_1x1')
        
        # skip connection
        fcn_layer4_skip = tf.add(fcn_layer7_up, fcn_layer4_1x1)

        # upsample
        fcn_layer4_up = tf.layers.conv2d_transpose(fcn_layer4_skip, num_classes,
                                                   kernel_size=4, strides=(2,2), padding='SAME',
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV_INI),
                                                   name='fcn_layer4_up')
    
        # 1x1 convolution of vgg layer 3
        fcn_layer3_1x1 = tf.layers.conv2d(layer3_out_scaled, num_classes,
                                          kernel_size=1, strides=(1,1), padding='SAME',
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV_INI),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
                                          name='fcn_layer3_1x1')

        # skip connection
        fcn_layer3_skip = tf.add(fcn_layer4_up, fcn_layer3_1x1)

        # upsample
        fcn_layer3_up = tf.layers.conv2d_transpose(fcn_layer3_skip, num_classes, 
                                                   kernel_size=16, strides=(8,8), padding='SAME',
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=STDDEV_INI), 
                                                   name='fcn_layer3_up')
        
        #tf.summary.histogram('fcn_layer7_1x1', fcn_layer7_1x1)
        #tf.summary.histogram('fcn_layer4_1x1', fcn_layer4_1x1) 
        #tf.summary.histogram('fcn_layer3_1x1', fcn_layer3_1x1)
        #tf.summary.histogram('fcn_layer7_up', fcn_layer7_up)
        #tf.summary.histogram('fcn_layer4_skip', fcn_layer4_skip)
        #tf.summary.histogram('fcn_layer4_up', fcn_layer4_up)
        #tf.summary.histogram('fcn_layer3_skip', fcn_layer3_skip)
        #tf.summary.histogram('fcn_layer3_up', fcn_layer3_up)
    
    return fcn_layer3_up
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    # Minimize error using cross entropy
    with tf.name_scope('Loss'):
       cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
       regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
       loss = cross_entropy_loss + 0.05 * regularization_loss
       
       #tf.summary.scalar('loss', loss)
    
    with tf.name_scope('Optimizer'):
       train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    
    # Merge all summaries into a single op and get op to write logs to Tensorboard
    #merged_summary_op = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter(LOG_PATH, graph=sess.graph)
    
    sess.run(tf.global_variables_initializer())
    
    print("Training begins")

    start_time = time.time()
    
    # Training cycle
    for epoch in range(epochs):
    
        # Loop over all batches
        for batch_images, batch_labels in get_batches_fn(batch_size):
            
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: batch_images,
                                          correct_label: batch_labels,
                                          keep_prob: KEEP_PROB, 
                                          learning_rate: LEARNING_RATE})

        print("Epoch: {}".format(epoch + 1), "/ {}".format(epochs), 
              " Loss: {:.3f}".format(loss),
              " Time: ", str(timedelta(seconds=(time.time() - start_time))))

        vars = 0
        for v in tf.all_variables():
            vars += np.prod(v.get_shape().as_list())
        print(vars)
    
    print("Optimization Finished!")
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = '/data'
    runs_dir = '/output/runs'

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')

        logits, train_op, loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
