import numpy as np
from modelBaseClassT import *


class vggModelT(modelBaseT):
    def __init__(self,gConfig,getdataClass):
        super(vggModelT,self).__init__(gConfig)
        self.input_shape = getdataClass.resizedshape
        self.get_net()
        self.merged = tf.summary.merge_all()

    def vgg_block(self,block_index, input_channels,num_convs, num_channels,conv_in,activation):
        vgg_name = 'vgg_block'+str(block_index)
        with tf.variable_scope(vgg_name):
            for i in range(num_convs):
                scop_name = 'conv'+str(i)
                #self.scopes.append(vgg_name+'/'+scop_name)
                with tf.name_scope(scop_name),tf.variable_scope(scop_name):
                    conv_filter = [3, 3, input_channels, num_channels]
                    conv_w = tf.get_variable(name='w', shape=conv_filter,
                                             initializer=self.get_initializer(self.initializer))
                    conv_b = tf.get_variable(name='b', shape=[num_channels],
                                             initializer=tf.constant_initializer(self.init_bias))
                    conv = tf.nn.conv2d(conv_in,conv_w,strides=[1,1,1,1],padding='SAME',name='conv')
                    conv_out = self.get_activation(activation)(conv + conv_b, name='conv_out')
                    conv_in = conv_out
                    input_channels = num_channels
            with tf.name_scope('pool'+str(block_index)):
                pool = tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',
                                      name='pool'+str(block_index))
        return pool,num_channels

    def get_net(self):
        super(vggModelT,self).get_net()
        conv_arch = self.gConfig['conv_arch']
        ratio = self.gConfig['ratio']
        small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
        #input_channels = self.gConfig['input_channels']
        #input_dim_x = self.gConfig['input_dim_x']
        #input_dim_y = self.gConfig['input_dim_y']
        input_channels,input_dim_x,input_dim_y = self.input_shape
        dense1_hiddens = self.gConfig['dense1_hiddens']  # // ratio#4096
        dense2_hiddens = self.gConfig['dense2_hiddens']  # // ratio#4096
        dense3_hiddens = self.gConfig['dense3_hiddens']  # 4096
        drop1_rate = self.gConfig['drop1_rate']  # 0.5
        drop2_rate = self.gConfig['drop2_rate']  # 0.5
        activation = self.gConfig['activation']
        class_num = self.gConfig['class_num']
        with tf.name_scope('input'):
            self.keep_prop = tf.placeholder(tf.float32, name="keep_prop")
            self.X_input = tf.placeholder(tf.float32, [None, input_channels, input_dim_x, input_dim_y], name='X_input')
            # self.X = tf.placeholder(tf.float32, [None, self.gConfig['xdim']*self.gConfig['ydim']], name="x")
            self.t_input = tf.placeholder(tf.int32, [None, ], name='t_input')
        with tf.name_scope('transpos'):
            self.X = tf.transpose(self.X_input, perm=[0, 2, 3, 1], name='X')
            self.t = tf.one_hot(self.t_input, class_num, axis=1, name='t')
            tf.summary.image('image', self.X)

        # 卷积层部分
        conv_in = self.X
        for block_index,(num_convs, num_channels) in enumerate(small_conv_arch):
            conv_in,input_channels=\
                self.vgg_block(block_index,input_channels,num_convs,num_channels,conv_in,activation=activation)

        with tf.name_scope('dense1'),tf.variable_scope('dense1'):
            #self.scopes.append('dense1')
            pool_out_dim = int(np.prod(conv_in.get_shape()[1:]))
            pool_flat = tf.reshape(conv_in, shape=[-1, pool_out_dim], name='pool_flattern')
            dense1_w = tf.get_variable(name='dense1_w', shape=[pool_out_dim, dense1_hiddens],
                                       initializer=self.get_initializer(self.initializer))
            dense1_b = tf.get_variable(name='dense1_b', shape=[dense1_hiddens],
                                       initializer=tf.constant_initializer(self.init_bias))
            dense1 = tf.nn.bias_add(tf.matmul(pool_flat, dense1_w), dense1_b, name='dense1')
            dense1_out = self.get_activation(activation)(dense1, name='dense1_out')
            drop1_out = tf.nn.dropout(dense1_out,(1-drop1_rate),name='drop1_out')

        with tf.name_scope('dense2'),tf.variable_scope('dense2'):
            #self.scopes.append('dense2')
            dense2_w = tf.get_variable(name='dense2_w', shape=[dense1_hiddens, dense2_hiddens],
                                       initializer=self.get_initializer(self.initializer))
            dense2_b = tf.get_variable(name='dense2_b', shape=[dense2_hiddens],
                                       initializer=tf.constant_initializer(self.init_bias))
            dense2 = tf.nn.bias_add(tf.matmul(drop1_out, dense2_w), dense2_b, name='dense2')
            dense2_out = self.get_activation(activation)(dense2, name='dense2_out')
            drop2_out = tf.nn.dropout(dense2_out,(1-drop2_rate),name='drop2_out')

        with tf.name_scope('dense3'),tf.variable_scope('dense3'):
            #self.scopes.append('dense3')
            dense3_w = tf.get_variable(name='dense3_w', shape=[dense2_hiddens, dense3_hiddens],
                                       initializer=self.get_initializer(self.initializer))
            dense3_b = tf.get_variable(name='dense3_b', shape=[dense3_hiddens],
                                       initializer=tf.constant_initializer(self.init_bias))
            dense3 = tf.nn.bias_add(tf.matmul(drop2_out, dense3_w), dense3_b, name='dense3')

        with tf.name_scope('softmax'):
            y_hat = tf.nn.softmax(dense3, name='y_hat')

        with tf.name_scope('loss'):
            self.loss = -tf.reduce_mean(self.t * tf.log(y_hat), name='loss')
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('evaluate'):
            is_correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(self.t, 1))
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.name_scope('train'):
            optimizer = self.get_optimizer(self.optimizer)
            self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)

    def run_train_loss_acc(self,X,y,keeps):
        if self.global_step.eval() == 0:
            self.debug_info(X,y)
        _, loss, acc, merged = self.session.run([self.train_step, self.loss, self.accuracy, self.merged],
                                                feed_dict={self.X_input: X, self.t_input: y,
                                                           self.keep_prop: float(keeps)})

        return loss,acc,merged

    def run_loss_acc(self,X,y,keeps=1.0):
        loss,acc = self.session.run([self.loss,self.accuracy],
                                   feed_dict={self.X_input:X,self.t_input:y,self.keep_prop:float(keeps)})
        return loss,acc

    def run_gradient(self,trainable_vars,X,y,keeps):
        grads = tf.gradients(self.loss, trainable_vars)
        grads_value = self.session.run(grads,feed_dict={self.X_input: X, self.t_input: y,
                                                   self.keep_prop: float(keeps)})
        return grads_value



def create_model(gConfig,ckpt_used,getdataClass):
    model=vggModelT(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model


