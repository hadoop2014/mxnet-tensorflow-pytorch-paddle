import numpy as np
from modelBaseClassP import *
from torch import optim

class alexnetModelP(modelBaseP):
    def __init__(self,gConfig,getdataClass):
        super(alexnetModelP,self).__init__(gConfig)
        self.resizedshape = getdataClass.resizedshape
        self.get_net()

    def get_net(self):
        #input_channels = self.gConfig['input_channels']
        #input_dim_x = self.gConfig['input_dim_x']
        #input_dim_y = self.gConfig['input_dim_y']
        input_channels,input_dim_x,input_dim_y = self.resizedshape
        conv1_channels = self.gConfig['conv1_channels']  # 96
        conv1_kernel_size = self.gConfig['conv1_kernel_size']  # 11
        conv1_strides = self.gConfig['conv1_striders']  # 4
        conv1_padding = self.gConfig['conv1_padding']  # 1
        pool1_size = self.gConfig['pool1_size']  # 3
        pool1_strides = self.gConfig['pool1_strides']  # 2
        pool1_padding = self.gConfig['pool1_padding'] #0
        conv2_channels = self.gConfig['conv2_channels']  # 256
        conv2_kernel_size = self.gConfig['conv2_kernel_size']  # 5
        conv2_strides = self.gConfig['conv2_strides']  # 1
        conv2_padding = self.gConfig['conv2_padding']  # 2
        pool2_size = self.gConfig['pool2_size']  # 3
        pool2_strides = self.gConfig['pool2_strides']  # 2
        pool2_padding = self.gConfig['pool2_padding']  #0
        conv3_channels = self.gConfig['conv3_channels']  # 384
        conv3_kernel_size = self.gConfig['conv3_kernel_size']  # 3
        conv3_strides = self.gConfig['conv3_strides']  # 1
        conv3_padding = self.gConfig['conv3_padding']  # 1
        conv4_channels = self.gConfig['conv4_channels']  # 384
        conv4_kernel_size = self.gConfig['conv4_kernel_size']  # 3
        conv4_strides = self.gConfig['conv4_strides']  # 1
        conv4_padding = self.gConfig['conv4_padding']  # 1
        conv5_channels = self.gConfig['conv5_channels']  # 256
        conv5_kernel_size = self.gConfig['conv5_kernel_size']  # 3
        conv5_strides = self.gConfig['conv5_strides']  # 1
        conv5_padding = self.gConfig['conv5_padding']  # 1
        pool3_size = self.gConfig['pool3_size']  # 3
        pool3_strides = self.gConfig['pool3_strides']  # 2
        pool3_padding = self.gConfig['pool3_padding'] #0
        dense1_hiddens = self.gConfig['dense1_hiddens']  # 4096
        drop1_rate = self.gConfig['drop1_rate']  # 0.5
        dense2_hiddens = self.gConfig['dense2_hiddens']  # 4096
        drop2_rate = self.gConfig['drop2_rate']  # 0.5
        dense3_hiddens = self.gConfig['dense3_hiddens']  # 10
        class_num = self.gConfig['class_num']#10
        activation = self.gConfig['activation'] #relu
        weight_initializer = self.get_initializer(self.initializer)
        bias_initializer = self.get_initializer('constant')

        with fluid.name_scope('input'):
            #self.X_input = fluid.layers.data(shape=[-1,input_channels*input_dim_x*input_dim_y],dtype='float32',
            #                                 stop_gradient=True,append_batch_size=False,
            #                                 name='X_input')
            self.X_input = fluid.layers.data(shape=[-1, input_channels , input_dim_x , input_dim_y],
                                             dtype='float32',
                                             stop_gradient=True,append_batch_size=False,
                                             name='X_input')
            self.t_input = fluid.layers.data(shape=[-1,],dtype='int64',
                                             stop_gradient=True,append_batch_size=False,
                                             name='t_input')
            self.is_test = fluid.layers.data(shape=[1],dtype='bool',stop_gradient=True,append_batch_size=False,
                                             name='is_test')
        with fluid.name_scope('transpos'):
            #self.X = fluid.layers.reshape(self.X_input,shape=[-1,input_channels,input_dim_x,input_dim_y],
            #                              name='X')
            self.X = self.X_input
            self.t = fluid.layers.reshape(self.t_input,shape=[-1,1],
                                          name='t')

        with fluid.name_scope('conv1'):#,tf.variable_scope('conv1'):
            conv1 = fluid.layers.conv2d(self.X,num_filters=conv1_channels,filter_size=conv1_kernel_size,
                                        stride=conv1_strides,padding=conv1_padding,
                                        act=self.get_activation(activation),
                                        param_attr=fluid.param_attr.ParamAttr(initializer=weight_initializer,
                                                                              name='conv1/weight'),
                                        bias_attr=fluid.param_attr.ParamAttr(initializer=bias_initializer,
                                                                             name='conv1/bias'),
                                        name='conv1')



        with fluid.name_scope('pool1'):
            pool1 = fluid.layers.pool2d(conv1,pool_size=pool1_size,pool_type='max',pool_stride=pool1_strides,
                                        pool_padding=pool1_padding,name='pool1')


        with fluid.name_scope('conv2'):#,tf.variable_scope('conv2'):
            conv2 = fluid.layers.conv2d(pool1, num_filters=conv2_channels, filter_size=conv2_kernel_size,
                                        stride=conv2_strides,padding=conv2_padding,
                                        act=self.get_activation(activation),
                                        param_attr=fluid.param_attr.ParamAttr(initializer=weight_initializer,
                                                                              name='conv2/weight'),
                                        bias_attr=fluid.param_attr.ParamAttr(initializer=bias_initializer,
                                                                             name='conv2/bias'),
                                        name='conv2')

        with fluid.name_scope('pool2'):
            pool2 = fluid.layers.pool2d(conv2, pool_size=pool2_size, pool_type='max', pool_stride=pool2_strides,
                                        pool_padding=pool2_padding, name='pool2')


        with fluid.name_scope('conv3'):#,tf.variable_scope('conv3'):
            conv3 = fluid.layers.conv2d(pool2, num_filters=conv3_channels, filter_size=conv3_kernel_size,
                                        stride=conv3_strides, padding=conv3_padding,
                                        act=self.get_activation(activation),
                                        param_attr=fluid.param_attr.ParamAttr(initializer=weight_initializer,
                                                                              name='conv3/weight'),
                                        bias_attr=fluid.param_attr.ParamAttr(initializer=bias_initializer,
                                                                             name='conv3/bias'),
                                        name='conv3')


        with fluid.name_scope('conv4'):
            conv4 = fluid.layers.conv2d(conv3, num_filters=conv4_channels, filter_size=conv4_kernel_size,
                                        stride=conv4_strides, padding=conv4_padding,
                                        act=self.get_activation(activation),
                                        param_attr=fluid.param_attr.ParamAttr(initializer=weight_initializer,
                                                                              name='conv4/weight'),
                                        bias_attr=fluid.param_attr.ParamAttr(initializer=bias_initializer,
                                                                             name='conv4/bias'),
                                        name='conv4')


        with fluid.name_scope('conv5'):
            conv5 = fluid.layers.conv2d(conv4, num_filters=conv5_channels, filter_size=conv5_kernel_size,
                                        stride=conv5_strides, padding=conv5_padding,
                                        act=self.get_activation(activation),
                                        param_attr=fluid.param_attr.ParamAttr(initializer=weight_initializer,
                                                                              name='conv5/weight'),
                                        bias_attr=fluid.param_attr.ParamAttr(initializer=bias_initializer,
                                                                             name='conv5/bias'),
                                        name='conv5')


        with fluid.name_scope('pool3'):
            pool3 = fluid.layers.pool2d(conv5, pool_size=pool3_size, pool_type='max', pool_stride=pool3_strides,
                                        pool_padding=pool3_padding, name='pool3')

        with fluid.name_scope('dense1'):
            dense1 = fluid.layers.fc(pool3,size=dense1_hiddens,num_flatten_dims=1,
                                     act=self.get_activation(activation),
                                     param_attr=fluid.param_attr.ParamAttr(initializer=weight_initializer,
                                                                           name='dense1/weight'),
                                     bias_attr=fluid.param_attr.ParamAttr(initializer=bias_initializer,
                                                                          name='dense1/bias'),
                                     name='dense1')
            drop1 = fluid.layers.dropout(dense1,dropout_prob=drop1_rate,#dropout_implementation='upscale_in_train',
                                             is_test=False,
                                             name='drop1')


        with fluid.name_scope('dense2'):
            dense2 = fluid.layers.fc(drop1,size=dense2_hiddens,
                                     act=self.get_activation(activation),
                                     param_attr=fluid.param_attr.ParamAttr(initializer=weight_initializer,
                                                                           name='dense2/weight'),
                                     bias_attr=fluid.param_attr.ParamAttr(initializer=bias_initializer,
                                                                          name='dense2/bias'),
                                     name='dense2')

        #with fluid.layers.control_flow.Switch() as switch:
        #    with switch.case(self.is_test ==
        #                     fluid.layers.fill_constant(shape=[1],dtype='bool',value=True)):
        #        drop2 = fluid.layers.dropout(dense2,dropout_prob=drop2_rate,
        #                                     #dropout_implementation='upscale_in_train',
        #                                     is_test=True,
        #                                     name='drop2')
        #    with switch.default():
            drop2 = fluid.layers.dropout(dense2, dropout_prob=drop2_rate,
                                             # dropout_implementation='upscale_in_train',
                                             is_test=False,
                                             name='drop2')

        with fluid.name_scope('dense3'):#,tf.variable_scope('dense3'):
            self.dense3 = fluid.layers.fc(drop2,size=class_num,
                                     act='softmax',
                                     param_attr=fluid.param_attr.ParamAttr(initializer=weight_initializer,
                                                                           name='dense3/weight'),
                                     bias_attr=fluid.param_attr.ParamAttr(initializer=bias_initializer,
                                                                          name='dense3/bias'),
                                     name='dense3')


        with fluid.name_scope('loss'):

            coss = fluid.layers.cross_entropy(self.dense3, self.t)
            self.loss = fluid.layers.mean(coss)

        with fluid.name_scope('evaluate'):
            self.accuracy = fluid.layers.accuracy(self.dense3,self.t)
        self.test_program = fluid.default_main_program().clone(for_test=True)

        with fluid.name_scope('train'):
            optimizer = self.get_optimizer(self.optimizer)
            self.train_step = optimizer.minimize(self.loss)

        #self.feeder = fluid.DataFeeder(feed_list=[self.X_input,self.t_input],place=self.places)
        self.main_program = fluid.default_main_program()

    def run_train_loss_acc(self,X,y,keeps):
        if self.global_step_value == 0:
            self.debug_info(X,y)
            fluid.io.save_inference_model(dirname=os.path.join(self.logging_directory,self.model_pb),
                                          feeded_var_names=[self.X_input.name,self.t_input.name],
                                          target_vars=[self.loss,self.accuracy,
                                                       self.learning_rate],
                                          executor=self.executor)
        loss, acc= self.executor.run(self.main_program,
                                     feed={self.X_input.name:X,self.t_input.name:y},
                                     fetch_list=[self.loss,self.accuracy])

        return loss,acc

    def run_loss_acc(self,X,y,keeps=1.0):
        loss,acc = self.executor.run(self.test_program,
                                     feed={self.X_input.name:X,self.t_input.name:y},
                                     fetch_list=[self.loss,self.accuracy])
        return loss,acc

    def run_gradient(self,trainable_vars,X,y,keeps):
        vars_value = []
        for var in trainable_vars :
            var_value = fluid.global_scope().find_var(var.name).get_tensor()
            vars_value.append(np.array(var_value))
        grads = fluid.gradients(self.loss, trainable_vars)
        grads_value = self.executor.run(self.main_program,
                                        feed={self.X_input.name:X,self.t_input.name:y},
                                        fetch_list=grads)
        return vars_value,grads_value

def create_model(gConfig,ckpt_used,getdataClass):
    model=alexnetModelP(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model


