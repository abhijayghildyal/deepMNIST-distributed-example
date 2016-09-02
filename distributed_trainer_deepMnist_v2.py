from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tempfile

import time
import numpy as np

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "","Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("folder_no", "", "save summary in this folder")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tf.app.flags.DEFINE_integer("epochs", "", "no of epochs")

tf.app.flags.DEFINE_string("weights_name", "", "weight file name")


FLAGS = tf.app.flags.FLAGS

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  
  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
    with tf.device("/job:ps/task:0"):
      global_step = tf.Variable(0)
      print global_step

  elif FLAGS.job_name == "worker":

    # Load data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      x = tf.placeholder(tf.float32, shape=[None, 784])
      y_ = tf.placeholder(tf.float32, shape=[None, 10])

      W = tf.Variable(tf.zeros([784,10]))
      b = tf.Variable(tf.zeros([10]))

      y = tf.nn.softmax(tf.matmul(x,W) + b)


      # Build model...
      # loss = 0.5

      ### deepMNIST code. Additions to the skeleton of distributed tensorflow code (taken from official documentation)###
      W_conv1 = weight_variable([5, 5, 1, 32])
      b_conv1 = bias_variable([32])


      x_image = tf.reshape(x, [-1,28,28,1])



      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)


      W_conv2 = weight_variable([5, 5, 32, 64])
      b_conv2 = bias_variable([64])

      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)


      W_fc1 = weight_variable([7 * 7 * 64, 1024])
      b_fc1 = bias_variable([1024])

      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


      keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


      W_fc2 = weight_variable([1024, 10])
      b_fc2 = bias_variable([10])

      y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

      tf.scalar_summary('cross entropy', cross_entropy)

      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      tf.scalar_summary('accuracy', accuracy)

      ########################################################################################################
      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          cross_entropy, global_step=global_step)
          # loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()
      merged = summary_op


    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.

      summaries_dir = 'summaries/'

      train_writer = tf.train.SummaryWriter(summaries_dir + FLAGS.folder_no + '/train', sess.graph)
      test_writer = tf.train.SummaryWriter(summaries_dir + FLAGS.folder_no + '/test', sess.graph)
      
      step=0
      time_start = time.time()
      while not sv.should_stop() and step < FLAGS.epochs:
        batch = mnist.train.next_batch(1000)
        train_feed = {x: batch[0], y_: batch[1], keep_prob: 0.5}        

        # Run a training step asynchronously.
        _, step = sess.run([train_op, global_step], feed_dict=train_feed)
        print "step: ",step
	
        summary, _ = sess.run([merged, train_op], train_feed)
        train_writer.add_summary(summary, step)
  
      time_end = time.time()
      training_time = time_end - time_start
      print("Training elapsed time: %f s" % training_time)
      
      np.savetxt("weights_dist_"+FLAGS.weights_name+".csv", W_fc2.eval(session=sess), delimiter=",")
	
      # Test
      for i in xrange(10):
        testSet = mnist.test.next_batch(50)
        summary, acc = sess.run([merged, accuracy], feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0})
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))

    # Ask for all the services to stop.
    sv.stop()

if  __name__ == "__main__":
  tf.app.run()