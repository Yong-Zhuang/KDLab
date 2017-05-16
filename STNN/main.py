###### import os.path
import time
import numpy as np
import tensorflow as tf
import layer_def as ld
import BasicConvLSTMCell
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '..../model',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 210,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 180,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 2000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', 1.0,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .01,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")
tf.app.flags.DEFINE_integer('training_epochs', 300,
                            """num of epochs""")
tf.app.flags.DEFINE_integer('width', 163,
                            """width""")
tf.app.flags.DEFINE_integer('hight', 138,
                            """height""")


data = np.load('../../no_crimes_perday_600_all.npy')
#data = data[:,18:88,58:148]

x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, FLAGS.hight, FLAGS.width, 1])
keep_prob = tf.placeholder("float")
def RNN(x):
    x_dropout = tf.nn.dropout(x, keep_prob)

    x_unwrap = []
    # create network
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([FLAGS.hight, FLAGS.width], [3,3], 1)
      new_state = cell.zero_state(FLAGS.batch_size, tf.float32) 

    # conv network
    for i in xrange(FLAGS.seq_length-1):
      # conv1
      if i < FLAGS.seq_start: #inputs, kernel_size, stride, num_features, idx
        #conv1 = ld.conv_layer(x_dropout[:,i,:,:,:], 3, 2, 8, "encode_1")
        conv1 = ld.transpose_conv_layer(x_dropout[:,i,:,:,:], 3, 1, 1, "decode_1")
      else:
        conv1 = ld.transpose_conv_layer(x_1, 3, 1, 1, "decode_1")
      y_0 = conv1
      # conv lstm cell 
      y_1, new_state = cell(y_0, new_state)
      # x_1
      x_1 = ld.conv_layer(y_1, 3, 1, 1, "encode_1", True)
      if i >= FLAGS.seq_start:
        x_unwrap.append(x_1)
      # set reuse to true after first go
      if i == 0:
        tf.get_variable_scope().reuse_variables()
    # pack them all together 
    x_unwrap = tf.stack(x_unwrap)
    x_unwrap = tf.transpose(x_unwrap, [1,0,2,3,4])
    return x_unwrap
pred = RNN(x)
# calc total loss (compare x_t to x_t+1)
loss = tf.nn.l2_loss(x[:,FLAGS.seq_start+1:,:,:,:] - pred[:,:,:,:,:])
#tf.scalar_summary('loss', loss)
tf.summary.scalar('loss', loss)
# training   
#train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
train_op = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(loss)
# List of all Variables
variables = tf.global_variables()

# Build a saver
saver = tf.train.Saver(variables)   

# Summary op
summary_op = tf.summary.merge_all()

# Build an initialization operation to run below.
init = tf.global_variables_initializer()
        
def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = range(0, len(inputs)-FLAGS.seq_length+1,7)
        #np.arange(len(inputs)-FLAGS.seq_length+1)
        #np.random.shuffle(indices)
    for start_idx in range(0, len(indices), batchsize):
        out = []
        if ((start_idx + batchsize)<=len(indices)):
            #print ('++++++111111111111++++++++'+str(start_idx))
            for start in indices[start_idx:start_idx + batchsize]:#(1797, 138, 163)
                out.append(inputs[start:start+FLAGS.seq_length])
        else:
            #print ('++++++222222222222++++++++'+str(start_idx))
            for start in indices[-batchsize:]:#(1797, 138, 163)
                out.append(inputs[start:start+FLAGS.seq_length])        
        out = np.array(out)
        yield out.reshape((-1, FLAGS.seq_length, inputs.shape[1], inputs.shape[2],1))  
        
Xtr = data[:-686]
Xva = data[-686:-343]
Xte = data[-343:]
checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
save_path = ''
# Launch the graph
def train():
    with tf.Session() as sess:
        sess.run(init)
        # Summary op
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)
        train_loss = [] 
        validation_loss = []  
        minloss = 0
        for epoch in range(FLAGS.training_epochs):          
            train_err = 0
            train_batches = 0
            start_time = time.time() 
            print ('********************training start********************')
            print (Xtr.shape)      
            for batch in iterate_minibatches(Xtr, FLAGS.batch_size, shuffle=True):
                inputs= batch  
                _, loss_r =sess.run([train_op,loss], feed_dict={x: inputs, keep_prob:FLAGS.keep_prob})
                train_err += loss_r
                train_batches += 1            
                print("Batch {} training loss:\t\t{:.6f}".format(
                    train_batches, train_err / train_batches))
            train_loss.append(train_err)
            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            print ('********************validation start********************')
            print (Xva.shape)      
            for batch in iterate_minibatches(Xva, FLAGS.batch_size, shuffle=True):
                inputs= batch  
                val_err += sess.run(loss, feed_dict={x: inputs, keep_prob:FLAGS.keep_prob})
                val_batches += 1       
                print("Batch {} validation loss:\t\t{:.6f}".format(
                    val_batches, val_err / val_batches))
            validation_loss.append(val_err)

            # Then we print the results for this epoch:        
            summary_str = sess.run(summary_op, feed_dict={x: inputs, keep_prob:FLAGS.keep_prob})
            summary_writer.add_summary(summary_str, epoch) 
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, FLAGS.training_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            assert not np.isnan(train_err), 'Model diverged with loss = NaN'
            if minloss > val_err:
                minloss = val_err
                save_path = saver.save(sess, checkpoint_path, global_step=epoch)  
                print("Model saved in file: %s" % save_path)
        loss_plot(validation_loss)
        print("Optimization Finished!")   
    
def test():
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # Restore model weights from previously saved model
        saver.restore(sess, save_path)
        #print("Model restored from file: %s" % save_path)

        test_batches = 0    
        print ('********************testing start********************')
        for batch in iterate_minibatches(Xte, FLAGS.batch_size, shuffle=True):
            inputs= batch  
            y_p = sess.run(pred, feed_dict={x: inputs, keep_prob:FLAGS.keep_prob})
            test_batches += 1        
    
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()
  #test()
if __name__ == '__main__':
  tf.app.run()
