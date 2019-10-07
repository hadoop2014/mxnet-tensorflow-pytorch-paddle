from modelBaseClassT import *

##胶囊网络的实现
class capsnetModelT(modelBaseT):
    def __init__(self,gConfig,getdataClass):
        super(capsnetModelT,self).__init__(gConfig)
        self.resizedshape = getdataClass.resizedshape
        self.classnum=getdataClass.classnum
        self.get_net()
        self.merged = tf.summary.merge_all()

    def get_net(self):
        # input_channels = self.gConfig['input_channels']
        # input_dim_x = self.gConfig['input_dim_x']
        # input_dim_y = self.gConfig['input_dim_y']
        input_channels, input_dim_x, input_dim_y = self.resizedshape
        conv1_channels = self.gConfig['conv1_channels']  # 256
        conv1_kernel_size = self.gConfig['conv1_kernel_size']  # 9
        conv1_strides = self.gConfig['conv1_strides']  # 1
        conv1_padding = self.gConfig['conv1_padding']  # 0
        conv1_dims = np.floor((input_dim_x - conv1_kernel_size + 2 * conv1_padding)/conv1_strides) + 1
        conv2_channels = self.gConfig['conv2_channels']  # 256
        conv2_kernel_size = self.gConfig['conv2_kernel_size']  # 9
        conv2_strides = self.gConfig['conv2_strides']  # 2
        conv2_padding = self.gConfig['conv2_padding']  # 0
        conv2_dims = np.floor((conv1_dims - conv2_kernel_size + 2 * conv2_padding)/conv2_strides) + 1
        caps1_caps = self.gConfig['caps1_caps']#1152
        caps1_dims = self.gConfig['caps1_dims']#8
        caps1_caps = int(conv2_dims * conv2_dims * conv2_channels / caps1_dims)
        print('caps1_caps=',caps1_caps)
        caps2_caps = self.gConfig['caps2_caps']#10
        caps2_caps = self.classnum  #替换为数据的实际类别数
        caps2_dims = self.gConfig['caps2_dims']#16
        routing_num = self.gConfig['routing_num']#2
        dense1_hiddens = self.gConfig['dense1_hiddens']#512
        dense1_hiddens = dense1_hiddens * input_channels
        dense2_hiddens = self.gConfig['dense2_hiddens']#1024
        # n_output = 28 * 28
        dense2_hiddens = dense2_hiddens * input_channels
        dense3_hiddens = self.gConfig['dense3_hiddens']#784
        dense3_hiddens = input_dim_x * input_dim_y*input_channels
        classnum = self.gConfig['class_num']
        classnum = self.classnum    #替换为数据的实际类别数
        mask_with_labels = self.gConfig['mask_with_labels']
        epsilon = self.gConfig['epsilon']
        m_plus = self.gConfig['m_plus']#0.9
        m_minus = self.gConfig['m_minus']#0.1
        caps_lambda = self.gConfig['caps_lambda']# 0.5
        caps_alpha = self.gConfig['caps_alpha']#0.0005
        activation = self.gConfig['activation']
        initializer = self.gConfig['initializer']
        weight_initializer = self.get_initializer(initializer)
        bias_initializer = self.get_initializer('constant')

        def safe_norm(s,axis = -1, epsilon = epsilon, keep_dims = False, name = None):
            with tf.name_scope(name, default_name='safe_norm'):
                squared_norm = tf.reduce_sum(tf.square(s), axis=axis,keepdims=keep_dims,name='squared_norm')
                return tf.sqrt(squared_norm + epsilon)

        def squash(s,axis = -1,epsilon = epsilon):
            s_sqr_norm = tf.reduce_sum(tf.square(s), axis=axis,keep_dims=True,name='s_sqr_norm')
            V = s_sqr_norm / (1. + s_sqr_norm)/tf.sqrt(s_sqr_norm + epsilon)
            return  V * s

        with tf.name_scope('input'):
            self.X_input = tf.placeholder(shape=[None,input_channels,input_dim_x,input_dim_y],dtype=tf.float32,name='X_input')
            self.t_input = tf.placeholder(tf.int64, [None,], name='t_input')  #维度[batch_size,10]
            #用于mask
            mask_with_labels = tf.placeholder_with_default(mask_with_labels,shape=(),name='mask_with_labels') #维度,任意

        with tf.name_scope('transpos'):
            self.X = tf.transpose(self.X_input, perm=[0, 2, 3, 1], name='X')
            batch_size = tf.shape(self.X)[0]
            print('X=,batch_size=',self.X,batch_size)
            self.t = tf.one_hot(self.t_input, classnum, axis=1, name='t')
            tf.summary.image('image', self.X)

        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(self.X,filters=conv1_channels,kernel_size=conv1_kernel_size,strides=conv1_strides,
                                     padding=self.get_padding(conv1_padding),
                                     activation=self.get_activation(activation),
                                     kernel_initializer=weight_initializer,
                                     bias_initializer=bias_initializer,
                                     name='conv1')

        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv2d(conv1,filters=conv2_channels,kernel_size=conv2_kernel_size,strides=conv2_strides,
                                     padding=self.get_padding(conv2_padding),
                                     activation=self.get_activation(activation),
                                     kernel_initializer=weight_initializer,
                                     bias_initializer=bias_initializer,
                                     name='conv2')               #[batch_size,6,6,256]


        with tf.variable_scope('caps1'):
            caps1_raw = tf.reshape(conv2,[batch_size,caps1_caps,caps1_dims],name='caps1_raw') #维度 [batch_size,1152,8]
            caps1_output = squash(caps1_raw)  #维度 [batch_size,1152,8]
            caps1_output_expanded = tf.expand_dims(caps1_output, -1, name='caps1_output_expanded') #维度[batch_size,1152,8,1]
            caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name='caps1_output_tile') #维度[batch_size,1152,1,8,1]
            caps1_output_tiled = tf.tile(caps1_output_tile,[1,1,caps2_caps,1,1],name='caps1_output_tiled') #维度[batch_size,1152,10,8,1]

            W = tf.get_variable(shape=[1,caps1_caps,caps2_caps,caps2_dims,caps1_dims],dtype=tf.float32,
                                initializer=self.get_initializer(initializer),name='W')
            W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name='W_tiled')  # 维度[batch_size,1152,10,16,8]
            #计算u hat
            caps2_predictd = tf.matmul(W_tiled,caps1_output_tiled,name='caps2_predicted') #维度[batch_size,1152,10,16,1]

        with tf.name_scope('caps2_routing'):
            #动态路由算法
            #用于动态路由算法
            raw_weights = tf.zeros([batch_size, caps1_caps, caps2_caps, 1, 1],
                                   dtype=tf.float32, name='raw_weights')  # 维度[batch_size, 1
            b = raw_weights  #维度[batch_size,1152,10,1,1]
            for i in range(routing_num):
                c = tf.nn.softmax(b, axis= 2,name='c') #维度[batch_size,1152,10,1,1]
                print("c.shape=",c)
                preds = tf.multiply(c, caps2_predictd,name='preds') #维度[batch_szie,1152,10,16,1]
                s = tf.reduce_sum(preds,axis=1,keepdims=True,name='s')  #维度[batch_size,1,10,16,1]
                print("s.shape=",s)
                vj = squash(s, axis=-2) #维度[batch_size,1,10,16,1]

                if i < routing_num -1:
                    vj_tiled = tf.tile(vj, [1,caps1_caps,1,1,1],name='vj_tiled') #维度[batch_size,1152,10,16,1]
                    agreement = tf.matmul(caps2_predictd, vj_tiled, transpose_a=True,name='agreement') #维度[batch_size,1152,10,1,1]
                    print("agreement.shape=",agreement)
                    b =tf.add(b, agreement,name='b')
            caps2_output = vj #维度[batch_size,1,10,16,1]

        with tf.name_scope('evaluate'):
            #估计实体出现的概率
            t_proba = safe_norm(caps2_output,axis=-2,name='t_proba') #维度[batch_size,1,10,1,1]
            t_proba_argmax = tf.argmax(t_proba,axis=2,name='t_proba_argmax') #维度[batch_size,1,1,1,1]
            t_pred = tf.squeeze(t_proba_argmax,axis=[1,2],name='t_pred') #维度[batch_size,1]
            correct = tf.equal(tf.argmax(self.t,dimension=1),t_pred,name='correct') #维度[batch_size,1]
            self.accuracy = tf.reduce_mean(tf.cast(correct,tf.float32),name='accuracy') #维度 1
            tf.summary.scalar('accuracy',self.accuracy)

        with tf.name_scope('margin_loss'):
            #计算边缘损失函数Margin Loss
            T = self.t
            caps2_output_norm = safe_norm(caps2_output,axis=-2,keep_dims=True,
                                          name='caps2_output_norm') #维度[batch_size,1,10,1,1]
            present_error_raw = tf.square(tf.maximum(0.,m_plus - caps2_output_norm),
                                          name='present_error_raw') #维度[batch_size,1,10,1,1]
            present_error = tf.reshape(present_error_raw,shape=[-1,caps2_caps],
                                       name='present_error') #维度[batch_size, 10]
            absent_error_raw = tf.square(tf.maximum(0.,caps2_output_norm - m_minus),
                                         name='absent_error_raw') #维度[batch_size,1,10,1,1]
            absent_error = tf.reshape(absent_error_raw, shape=[-1,caps2_caps],
                                      name='absent_error') #维度[batch_ize,10]
            L = tf.add(tf.cast(T,tf.float32) * present_error, caps_lambda * tf.cast((1 - T),tf.float32) * absent_error,
                       name='L') #维度[batch_size,10]
            self.margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name='margin_loss')
            tf.summary.scalar('margin_loss',self.margin_loss)

        #以下用于从结果中重构输入图片,重构损失函数
        with tf.name_scope('mask'):
            #Mask机制
            reconstruction_targets = tf.cond(mask_with_labels,lambda:tf.argmax(self.t,1),lambda :t_pred,
                                             name='reconstruction_targets') #维度[batch_size,1]
            reconstruction_mask = tf.one_hot(reconstruction_targets,depth=caps2_caps,
                                             name='reconstruction_mask') #维度[batch_size,10]
            reconstruction_mask_reshaped = tf.reshape(reconstruction_mask,[-1,1,caps2_caps,1,1],
                                                      name='reconstruction_mask_reshaped') #维度[batch_size,1,10,1,1]

            caps2_output_masked = tf.multiply(caps2_output,reconstruction_mask_reshaped,
                                              name='caps2_output_masked') #维度[batch_size,1,10,16,1]

        with tf.variable_scope('decoder'):
            decode_input = tf.reshape(caps2_output_masked,[-1,caps2_caps * caps2_dims],
                                      name='decode_input') #维度[batch_size, 160]
            dense1 = tf.layers.dense(decode_input, dense1_hiddens, activation=self.get_activation(activation),
                                      name='dense1') #维度[batch_size,512]
            dense2 = tf.layers.dense(dense1,dense2_hiddens,activation =self.get_activation(activation),
                                      name='dense2') #维度[batch_size,1024]
            decoder_output = tf.layers.dense(dense2, dense3_hiddens, activation=self.get_activation('sigmoid'),
                                             name="decoder_output") #维度[batch_size,784]
            print(decoder_output)
            x_out = tf.reshape(decoder_output, [-1, input_dim_x,input_dim_y, input_channels],
                               name='x_out')  # 维度[batch_size,28,28,1]
            tf.summary.image("reconstruct image",x_out)
            
        with tf.name_scope('reconstruction_loss'):
            squared_difference = tf.square(tf.reshape(self.X,
                                                      shape=[-1,input_dim_x*input_dim_y*input_channels]) - decoder_output,
                                           name="squared_difference")  # 维度[batch_size, 784]
            self.reconstruction_loss = caps_alpha*tf.reduce_sum(squared_difference,name='reconstruction_loss')   #维度 1
            tf.summary.scalar('reconstruction_loss',self.reconstruction_loss)

        with tf.name_scope('final_loss'):
            self.loss = tf.add(self.margin_loss, self.reconstruction_loss,name='loss')  #采用loss目的是为了和框架保持一致
            tf.summary.scalar('loss',self.loss)

        with tf.name_scope('train'):
            optimizer = self.get_optimizer(optimizer=self.optimizer)
            self.train_step = optimizer.minimize(self.loss,global_step=self.global_step,name='training_op')

        with tf.name_scope('initiation'):
            init_op = tf.global_variables_initializer()

    def run_train_loss_acc(self, X, y, keeps):
        if self.global_step.eval() == 0:
            self.debug_info(X, y)
        _, loss, acc, merged = self.session.run([self.train_step, self.loss, self.accuracy, self.merged],
                                                feed_dict={self.X_input: X, self.t_input: y})

        return loss, acc, merged

    def run_eval_loss_acc(self, X, y, keeps=1.0):
        loss, acc = self.session.run([self.loss, self.accuracy],
                                     feed_dict={self.X_input: X, self.t_input: y})
        return loss, acc

    def run_gradient(self, trainable_vars, X, y, keeps):
        grads = tf.gradients(self.loss, trainable_vars)
        grads_value = self.session.run(grads, feed_dict={self.X_input: X, self.t_input: y})
        return grads_value

def create_model(gConfig,ckpt_used,getdataClass):
    model=capsnetModelT(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model











