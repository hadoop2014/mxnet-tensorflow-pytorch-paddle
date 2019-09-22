from modelBaseClassT import *

class lenetModelT(modelBaseT):
    def __init__(self,gConfig,getdataClass):
        super(lenetModelT,self).__init__(gConfig)
        self.resizedshape = getdataClass.resizedshape
        self.get_net()
        self.merged = tf.summary.merge_all()

    def get_net(self):
        super(lenetModelT,self).get_net()
        input_dim_x = self.gConfig['input_dim_x']#28
        input_dim_y = self.gConfig['input_dim_y']#28
        input_channels = self.gConfig['input_channels'] #1
        input_channels,input_dim_x,input_dim_y = self.resizedshape
        activation = self.gConfig['activation']  # sigmoid
        conv1_channels = self.gConfig['conv1_channels']  # 6
        conv1_kernel_size = self.gConfig['conv1_kernel_size']  # 5
        conv1_strides = self.gConfig['conv1_strides']  # 1
        conv1_padding = self.gConfig['conv1_padding']  # 0
        pool1_size = self.gConfig['pool1_size']  # 2
        pool1_strides = self.gConfig['pool1_strides']  # 2
        pool1_padding = self.gConfig['pool1_padding']  # 0
        conv2_channels = self.gConfig['conv2_channels']  # 16
        conv2_kernel_size = self.gConfig['conv2_kernel_size']  # 5
        conv2_strides = self.gConfig['conv2_striders']  # 1
        conv2_padding = self.gConfig['conv2_padding']  # 0
        pool2_size = self.gConfig['pool2_size']  # 2
        pool2_padding = self.gConfig['pool2_padding']  # 0
        pool2_strides = self.gConfig['pool2_strides']  # 2
        dense1_hiddens = self.gConfig['dense1_hiddens']  # 120
        dense2_hiddens = self.gConfig['dense2_hiddens']  # 84
        dense3_hiddens = self.gConfig['dense3_hiddens']  # 10
        class_num = self.gConfig['class_num'] #10

        with tf.name_scope('input'):
            self.keep_prop = tf.placeholder(tf.float32, name="keep_prop")
            self.X_input = tf.placeholder(tf.float32,[None,input_channels,input_dim_x,input_dim_y],name='X_input')
            self.t_input = tf.placeholder(tf.int32, [None,],name = 't_input')
        with tf.name_scope('transpos'):
            self.X = tf.transpose(self.X_input,perm=[0,2,3,1],name='X')
            self.t = tf.one_hot(self.t_input,class_num,axis=1,name='t')
            tf.summary.image('image', self.X)
        with tf.name_scope('conv1'),tf.variable_scope('conv1'):
            conv1_filter=[conv1_kernel_size,conv1_kernel_size,input_channels,conv1_channels]
            conv1_w = tf.get_variable(name='conv1_w', shape=conv1_filter,
                                      initializer=self.get_initializer(self.initializer))
            conv1_b = tf.get_variable(name='conv1_b',shape=[conv1_channels],
                                      initializer=tf.constant_initializer(self.init_bias))
            tf.summary.histogram('conv1_w', conv1_w)
            tf.summary.histogram('conv1_b',conv1_b)
            conv1 = tf.nn.conv2d(self.X, conv1_w, strides=[1, conv1_strides, conv1_strides, 1],
                                 padding=self.get_padding(conv1_padding), name='conv1')
            conv1_out = self.get_activation(activation)(conv1 + conv1_b, name='conv1_out')
        with tf.name_scope('pool1'):
            pool1 = tf.nn.max_pool(conv1_out, ksize=[1, pool1_size, pool1_size, 1],
                                   strides=[1, pool1_strides, pool1_strides, 1],
                                   padding=self.get_padding(pool1_padding),
                                   name='pool1')

        with tf.name_scope('conv2'),tf.variable_scope('conv2'):
            conv2_filter=[conv2_kernel_size,conv2_kernel_size,conv1_channels,conv2_channels]
            conv2_w = tf.get_variable(name='conv2_w', shape=conv2_filter, initializer=self.get_initializer(self.initializer))
            conv2_b = tf.get_variable(name='conv2_b',shape=[conv2_channels],initializer=tf.constant_initializer(self.init_bias))
            conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1,conv2_strides,conv2_strides,1],
                                 padding=self.get_padding(conv2_padding), name='conv2')
            conv2_out = self.get_activation(activation)(conv2 + conv2_b, name='conv2_out')
        with tf.name_scope('pool2'):
            pool2 = tf.nn.max_pool(conv2_out, ksize=[1,pool2_size,pool2_size,1],
                                   strides=[1,pool2_strides,pool2_strides,1],
                                   padding=self.get_padding(pool2_padding))
            pool2_out_dim =  int(np.prod(pool2.get_shape()[1:]))
            pool2_flat = tf.reshape(pool2, shape=[-1, pool2_out_dim], name='pool2_flattern')
        with tf.name_scope('dense1'),tf.variable_scope('dense1'):
            dense1_w = tf.get_variable(name='dense1_w', shape=[pool2_out_dim,dense1_hiddens],
                                       initializer=self.get_initializer(self.initializer))
            dense1_b = tf.get_variable(name='dense1_b',shape=[dense1_hiddens],
                                       initializer=tf.constant_initializer(self.init_bias))

            dense1 = tf.nn.bias_add(tf.matmul(pool2_flat,dense1_w),dense1_b,name='dense1')
            dense1_out = self.get_activation(activation)(dense1, name='dense1_out')
        with tf.name_scope('dense2'),tf.variable_scope('dense2'):
            dense2_w = tf.get_variable(name='dense2_w', shape=[dense1_hiddens,dense2_hiddens],
                                       initializer=self.get_initializer(self.initializer))
            dense2_b = tf.get_variable(name='dense2_b',shape=[dense2_hiddens],
                                       initializer=tf.constant_initializer(self.init_bias))
            dense2 = tf.nn.bias_add(tf.matmul(dense1_out,dense2_w),dense2_b,name='dense2')
            dense2_out = self.get_activation(activation)(dense2, name='dense2_out')
        with tf.name_scope('dense3'),tf.variable_scope('dense3'):
            dense3_w = tf.get_variable(name='dense3_w', shape=[dense2_hiddens,dense3_hiddens],
                                       initializer=self.get_initializer(self.initializer))
            dense3_b = tf.get_variable(name='dense3_b',shape=[dense3_hiddens],
                                       initializer=tf.constant_initializer(self.init_bias))
            dense3 = tf.nn.bias_add(tf.matmul(dense2_out,dense3_w),dense3_b,name='dense3')
        with tf.name_scope('softmax'):
            y_hat = tf.nn.softmax(dense3,name='y_hat')

        with tf.name_scope('loss'):
            self.loss = -tf.reduce_mean(self.t * tf.log(y_hat), name='loss')
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('evaluate'):
            is_correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(self.t, 1))
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32),name='accuracy')
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.name_scope('train'):
            optimizer = self.get_optimizer(self.optimizer)
            self.train_step = optimizer.minimize(self.loss,global_step=self.global_step)


    def run_train_loss_acc(self,X,y,keeps):
        if self.global_step.eval() == 0:
            self.debug_info(X,y)
        _, loss, acc, merged = self.session.run([self.train_step, self.loss, self.accuracy, self.merged],
                                                feed_dict={self.X_input: X, self.t_input: y,
                                                           self.keep_prop: float(keeps)})
        return loss,acc,merged

    def run_eval_loss_acc(self, X, y, keeps=1.0):
        loss,acc = self.session.run([self.loss,self.accuracy],
                                   feed_dict={self.X_input:X,self.t_input:y,self.keep_prop:float(keeps)})
        return loss,acc

    def run_gradient(self,trainable_vars,X,y,keeps):
        grads = tf.gradients(self.loss, trainable_vars)
        grads_value = self.session.run(grads,feed_dict={self.X_input: X, self.t_input: y,
                                                   self.keep_prop: float(keeps)})
        return grads_value

def create_model(gConfig,ckpt_used,getdataClass):
    model=lenetModelT(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model


