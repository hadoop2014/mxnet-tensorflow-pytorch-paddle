from modelBaseClassT import *

class resnetModelT(modelBaseT):
    def __init__(self,gConfig,getdataClass):
        super(resnetModelT,self).__init__(gConfig)
        self.resizedshape = getdataClass.resizedshape
        self.classnum = getdataClass.classnum
        self.get_net()
        self.merged = tf.summary.merge_all()

    def residual_block(self,block_index, residual_index,input_channels,num_channels,conv_in,
                       activation,isTraining,use_1x1conv=False,strides=1):
        residual_name = 'block'+str(block_index) + '/residual'+str(residual_index)
        with tf.variable_scope(residual_name):
            with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
                conv1_filter = [3, 3, input_channels, num_channels]
                conv1_w = tf.get_variable(name='w', shape=conv1_filter,
                                          initializer=self.get_initializer(self.initializer))
                conv1_b = tf.get_variable(name='b', shape=[num_channels],
                                         initializer=tf.constant_initializer(self.init_bias))
                conv1 = tf.nn.conv2d(conv_in,conv1_w,strides=[1,strides,strides,1],padding='SAME',name='conv')
                conv1_out = tf.add(conv1 , conv1_b, name='conv_out')

            bn1 = tf.layers.batch_normalization(conv1_out,training=isTraining,name='bn1')
            bn1_out = self.get_activation(activation)(bn1,name='bn1_out')
            with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
                conv2_filter = [3, 3, num_channels, num_channels]
                conv2_w = tf.get_variable(name='w', shape=conv2_filter,
                                          initializer=self.get_initializer(self.initializer))
                conv2_b = tf.get_variable(name='b', shape=[num_channels],
                                          initializer=tf.constant_initializer(self.init_bias))
                conv2 = tf.nn.conv2d(bn1_out, conv2_w, strides=[1, 1, 1, 1], padding='SAME', name='conv')
                conv2_out = tf.add(conv2 ,conv2_b, name='conv_out')
            bn2 = tf.layers.batch_normalization(conv2_out,training=isTraining,name='bn2')

            if use_1x1conv == True:
                with tf.variable_scope('conv3',reuse=tf.AUTO_REUSE):
                    conv3_filter = [1, 1, input_channels, num_channels]
                    conv3_w = tf.get_variable(name='w', shape=conv3_filter,
                                              initializer=self.get_initializer(self.initializer))
                    conv3_b = tf.get_variable(name='b', shape=[num_channels],
                                              initializer=tf.constant_initializer(self.init_bias))
                    conv3 = tf.nn.conv2d(conv_in, conv3_w, strides=[1, strides, strides, 1], padding='SAME', name='conv')
                    conv3_out = tf.add(conv3 , conv3_b, name='conv_out')

                residual_out = self.get_activation(activation)(conv3_out+bn2,name='residual_out')
                return residual_out,num_channels

            residual_out = self.get_activation(activation)(bn2+conv_in,name='residual_out')
        return residual_out,num_channels

    def rest_block(self,block_index, input_channels, num_residuals,num_channels,conv_in,
                   activation,first_block,isTraining):
        for residual_index in range(num_residuals):
            if residual_index == 0 and first_block == False:
                conv_in,input_channels=self.residual_block(block_index,residual_index,input_channels,num_channels,
                                                           conv_in,
                                                           activation,
                                                           isTraining,
                                                           use_1x1conv=True,
                                                           strides=2)
            else:
                conv_in,input_channels=self.residual_block(block_index,residual_index,input_channels,num_channels,
                                                           conv_in,
                                                           activation,
                                                           isTraining)
        return conv_in,input_channels

    def get_net(self):
        residual_arch = self.gConfig['residual_arch']
        ratio = self.gConfig['ratio']
        small_residual_arch = [(pair[0], pair[1] // ratio) for pair in residual_arch]
        input_channels = self.gConfig['input_channels']
        input_dim_x = self.gConfig['input_dim_x']
        input_dim_y = self.gConfig['input_dim_y']
        input_channels,input_dim_x,input_dim_y = self.resizedshape
        conv1_channels = self.gConfig['conv1_channels'] // ratio
        conv1_kernel_size = self.gConfig['conv1_kernel_size']
        conv1_strides = self.gConfig['conv1_strides']
        conv1_padding = self.gConfig['conv1_padding']
        pool1_size = self.gConfig['pool1_size']
        pool1_strides = self.gConfig['pool1_strides']
        pool1_padding = self.gConfig['pool1_padding']
        dense1_hiddens = self.gConfig['dense1_hiddens']
        class_num = self.gConfig['class_num']
        classnum = self.classnum
        activation = self.gConfig['activation']

        with tf.name_scope('input'):
            self.keep_prop = tf.placeholder(tf.float32, name="keep_prop")
            self.X_input = tf.placeholder(tf.float32, [None, input_channels, input_dim_x, input_dim_y], name='X_input')
            self.t_input = tf.placeholder(tf.int32, [None, ], name='t_input')
            self.isTraining = tf.placeholder(tf.bool,name='isTraining')
        with tf.name_scope('transpos'):
            self.X = tf.transpose(self.X_input, perm=[0, 2, 3, 1], name='X')
            self.t = tf.one_hot(self.t_input, classnum, axis=1, name='t')
            tf.summary.image('image', self.X)

        # 卷积层部分
        with tf.variable_scope('conv1'):
            conv1_filter = [conv1_kernel_size, conv1_kernel_size, input_channels, conv1_channels]
            conv1_w = tf.get_variable(name='conv1_w', shape=conv1_filter,
                                      initializer=self.get_initializer(self.initializer))
            conv1_b = tf.get_variable(name='conv1_b', shape=[conv1_channels],
                                      initializer=tf.constant_initializer(self.init_bias))
            tf.summary.histogram('conv1_w', conv1_w)
            tf.summary.histogram('conv1_b', conv1_b)
            conv1 = tf.nn.conv2d(self.X, conv1_w, strides=[1, conv1_strides, conv1_strides, 1],
                                 padding=self.get_padding(conv1_padding), name='conv1')
            conv1_out = self.get_activation(activation)(conv1 + conv1_b, name='conv1_out')
        with tf.variable_scope('batch_norm1'):
            bn1 = tf.layers.batch_normalization(conv1_out,training=self.isTraining,name='bn1')
            bn1_out = self.get_activation(activation)(bn1,name='activation')
        with tf.name_scope('pool1'):
            pool1 = tf.nn.max_pool(bn1_out,ksize=[1,pool1_size,pool1_size,1],
                                   strides=[1,pool1_strides,pool1_strides,1],
                                   padding=self.get_padding(pool1_padding),
                                   name='pool1')

        first_block = True
        conv_in = pool1
        input_channels = conv1_channels
        for block_index,(num_residuals, num_channels) in enumerate(small_residual_arch):
            conv_in,input_channels=\
                self.rest_block(block_index,input_channels,num_residuals,num_channels,conv_in,
                                activation,first_block,self.isTraining)
            first_block = False

        with tf.name_scope('global_pool'):
            global_pool = tf.reduce_mean(conv_in,axis=[1,2],keepdims=True,name='global_pool')

        with tf.variable_scope('dense1'):
            pool_squeeze = tf.squeeze(global_pool,name='pool_squeeze')
            dense1_w = tf.get_variable(name='dense1_w', shape=[input_channels, classnum],
                                       initializer=self.get_initializer(self.initializer))
            dense1_b = tf.get_variable(name='dense1_b', shape=[classnum],
                                       initializer=tf.constant_initializer(self.init_bias))
            dense1 = tf.nn.bias_add(tf.matmul(pool_squeeze, dense1_w), dense1_b, name='dense1')

        with tf.name_scope('softmax'):
            y_hat = tf.nn.softmax(dense1, name='y_hat')

        with tf.name_scope('loss'):
            self.loss = -tf.reduce_mean(self.t * tf.log(y_hat), name='loss')
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('evaluate'):
            is_correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(self.t, 1))
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.name_scope('train'):
            optimizer = self.get_optimizer(self.optimizer)
            #启用batch_norm时必须用到
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)

    def run_train_loss_acc(self,X,y,keeps):
        if self.global_step.eval() == 0:
            self.debug_info(X,y)
        _, loss, acc, merged = self.session.run([self.train_step, self.loss, self.accuracy, self.merged],
                                                feed_dict={self.X_input: X, self.t_input: y,
                                                           self.keep_prop: float(keeps),
                                                           self.isTraining:True})

        return loss,acc,merged

    def run_eval_loss_acc(self, X, y, keeps=1.0):
        loss,acc = self.session.run([self.loss,self.accuracy],
                                    feed_dict={self.X_input:X,
                                               self.t_input:y,
                                               self.keep_prop:float(keeps),
                                               self.isTraining:False})
        return loss,acc

    def run_gradient(self,trainable_vars,X,y,keeps):
        grads = tf.gradients(self.loss, trainable_vars)
        grads_value = self.session.run(grads,feed_dict={self.X_input: X,
                                                        self.t_input: y,
                                                        self.keep_prop: float(keeps),
                                                        self.isTraining:False})
        return grads_value

def create_model(gConfig,ckpt_used,getdataClass):
    model=resnetModelT(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model


