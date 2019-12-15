#使用tensorflow2.0, 采用keras框架
import tensorflow as tf
from tensorflow import keras
#import tensorflow as tf
from modelBaseClass import *
#from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python import debug as tfdbg
import numpy as np

#深度学习模型的基类
class modelBaseK(modelBase):
    def __init__(self,gConfig):
        super(modelBaseK,self).__init__(gConfig)
        self.learning_rate_value = self.gConfig['learning_rate']
        self.learning_rate_decay_factor = self.gConfig['learning_rate_decay_factor']
        self.decay_steps = self.gConfig['decay_steps']
        self.init_sigma = self.gConfig['init_sigma']
        self.init_bias = self.gConfig['init_bias']
        self.momentum = self.gConfig['momentum']
        self.max_to_keep = self.gConfig['max_to_keep']
        self.max_queue = self.gConfig['max_queue']
        #self.session =self.get_session(self.gConfig['ctx'])
        self.initializer = self.gConfig['initializer']
        self.optimizer = self.gConfig['optimizer']
        self.tfdbgIsOn = self.gConfig['tfdbgIsOn'.lower()]
        self.keeps =self.gConfig['keeps']
        self.log_savefile='.'.join([self.get_model_name(gConfig),'log'])
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.Variable(float(self.learning_rate_value), trainable=False, name='learning_rate')
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate *
                                                                    self.learning_rate_decay_factor,
                                                                    name='learning_rate_decay')
            #self.learning_rate = tf.train.exponential_decay(self.learning_rate_value,self.global_step,self.decay_steps,
            #                                                self.learning_rate_decay_factor,staircase=True)
        self.net = keras.Sequential()

    def get_net(self):
        return

    '''def get_session(self,ctx):
        assert ctx in self.gConfig['ctxlist'], 'ctx(%s) is invalid,it must one of %s' % \
                                                               (ctx, self.gConfig['ctxlist'])
        if ctx == 'gpu':
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'
            config.gpu_options.allow_growth = True
            #config.gpu_options.per_process_gpu_memory_fraction = 0.1
            Session = tf.Session(config=config)
        else:
            Session = tf.Session()
        return Session'''

    def get_initializer(self, initializer):
        assert initializer in self.gConfig['initializerlist'],'initializer(%s) is invalid,it must one of %s' % \
                                                    (initializer, self.gConfig['initializerlist'])
        if initializer == 'normal':
            #return tf.truncated_normal_initializer(stddev=self.init_sigma)
            return keras.initializers.TruncatedNormal(stddev=self.init_sigma)
        elif initializer == 'xavier':
            #return tf.glorot_normal_initializer()
            return keras.initializers.GlorotNormal()
        elif initializer == 'kaiming':
            #何凯明初始化法
            #return tf.glorot_uniform_initializer()
            return keras.initializers.he_uniform()
        elif initializer == 'constant':
            #return tf.constant_initializer(self.init_bias)
            return keras.initializers.identity(self.init_bias)
        else:
            return None

    def get_optimizer(self,optimizer):
        assert optimizer in self.gConfig['optimizerlist'], 'optimizer(%s) is invalid,it must one of %s' % \
                                                               (optimizer, self.gConfig['optimizerlist'])
        if optimizer == 'sgd':
            #return  tf.train.GradientDescentOptimizer(self.learning_rate)
            return keras.optimizers.SGD(self.learning_rate)
        elif optimizer == 'adadelta':
            #return  tf.train.AdadeltaOptimizer(self.learning_rate)
            return keras.optimizers.Adadelta(self.learning_rate)
        elif optimizer == 'rmsprop':
            #return  tf.train.RMSPropOptimizer(self.learning_rate)
            return keras.optimizers.RMSprop(self.learning_rate)
        elif optimizer == 'adam':
            #return tf.train.AdamOptimizer(self.learning_rate)
            return keras.optimizers.Adam(self.learning_rate)
        elif optimizer == 'adagrad':
            #return tf.train.AdagradOptimizer(self.learning_rate)
            return keras.optimizers.Adagrad(self.learning_rate)
        elif optimizer == 'momentum':
            #return tf.train.MomentumOptimizer(self.learning_rate,momentum=self.momentum)
            return keras.optimizers.SGD(learning_rate=self.learning_rate,momentum=self.momentum)
        return None

    def get_padding(self):
        #把padding数字转化为tensorflow.nn.conv2d的函数入参
        def get_padding_iner(padding):
            if padding == 0:
                return 'VALID'
            else:
                return 'SAME'
        return get_padding_iner

    def get_activation(self, activation='relu'):
        assert activation in self.gConfig['activationlist'], 'activation(%s) is invalid,it must one of %s' % \
                                                    (activation, self.gConfig['activationlist'])
        if activation == 'sigmoid':
            #return tf.nn.sigmoid
            return keras.layers.Activation('sigmoid')
        elif activation == 'relu':
            #return tf.nn.relu
            return keras.layers.Activation('relu')

    def get_context(self):
        devices = self.session.list_devices()
        return devices
        #return [device.name for device in devices]

    def get_learningrate(self):
        return self.learning_rate.eval(session=self.session)


    def get_globalstep(self):
        return self.global_step.eval(session=self.session)


    def saveCheckpoint(self):
        #checkpoint_path = os.path.join(self.working_directory, self.checkpoint_filename)
        #self.saver.save(self.session, checkpoint_path, global_step=self.global_step.eval())
        self.net.save(self.model_savefile)
        tf.train.Checkpoint(x=self.global_step).save(self.working_directory)

    def train(self,model_eval,getdataClass,gConfig,num_epochs):
        #with self.session as session:
        #    if self.tfdbgIsOn == True:
        #        self.session = tfdbg.LocalCLIDebugWrapperSession(session,ui_type='readline')
        #        self.session.add_tensor_filter('has_nif_or_nan',tfdbg.has_inf_or_nan)
        for epoch in range(num_epochs):
            self.run_epoch(getdataClass, epoch)

        return self.losses_train,self.acces_train,self.losses_valid,self.acces_valid,\
               self.losses_test,self.acces_test

    def debug_info(self,*args):
        '''if len(args) == 0:
            trainabled_vars = tf.trainable_variables()
            for trainabled_var in trainabled_vars:
                print('\tdebug:%s' % trainabled_var.name,
                      '\tshape=', trainabled_var.shape)
            return
        (X,y) = args
        if self.debugIsOn == False:
            return
        self.debug(X,y)'''
        if self.debugIsOn == False:
            return
        if len(args) > 0:
            print('debug:%s'%args[0])
            return
        for (key,layer) in self.net.named_children():
            self.debug(layer,':'.join([key,layer._get_name()]))
        print('\n')
        return

    def debug(self,X,y):
        trainable_vars = tf.trainable_variables()
        grads = self.run_gradient(trainable_vars,X,y,self.keeps)
        for trainable_var,grad in zip(trainable_vars,grads):
            print('\tdebug(%d):%s'%(self.global_step.eval(),trainable_var.name),
                  '\tshape=',trainable_var.shape,
                  '\tdata.mean=%.6f'%trainable_var.eval().mean(),
                  '\tgrad.mean=%.6f' % grad.mean(),
                  '\tdata.std=%.6f'%trainable_var.eval().std(),
                  '\tgrad.std=%.6f'%grad.std())

    def run_gradient(self,grads,X,y,keeps):
        grads_value = None
        return grads_value

    def run_train_loss_acc(self,X,y,keeps):
        loss,acc,merged = None,None,None
        return loss,acc,merged

    def run_eval_loss_acc(self, X, y, keeps=1.0):
        loss,acc = None,None
        return loss,acc

    def run_learning_rate_decay(self,acc_train):
        if len(self.acces_train) > 2 and acc_train < min(self.acces_train[-3:]):
            self.session.run(self.learning_rate_decay_op)
            print("\n execute learning rate decay(learning_rate=%f)\n" % self.learning_rate.eval())

    def train_loss_acc(self,data_iter):
        acc_sum = 0
        loss_sum =0
        num = 0
        for X, y in data_iter:
            try:
                X = X.asnumpy()
                y = y.asnumpy()
            except:
                X = np.array(X)
                y = np.array(y)
            #if self.global_step.eval()  == 0:
            #    self.debug_info(X, y)
            loss, acc = self.run_train_loss_acc(X, y,self.keeps)
            if self.global_step == 0 or self.global_step == 1:
                self.debug_info()
            batch_size = y.size
            loss_sum += loss * batch_size
            acc_sum += acc * batch_size
            num += batch_size
            self.global_step += tf.constant([1])
            self.writer.add_summary(global_step=self.global_step.eval())
        return loss_sum / num, acc_sum / num

    def evaluate_loss_acc(self, data_iter):
        acc_sum = 0
        loss_sum =0
        num = 0
        for X, y in data_iter:
            try:
                #由mxnet框架读入的数据通过该方法转换
                X = X.asnumpy()
                y = y.asnumpy()
            except:
                #由其他框架读入的数据通过numpy转换
                X = np.array(X)
                y = np.array(y)
            loss,acc = self.run_eval_loss_acc(X, y)
            batch_size = y.size
            loss_sum += loss * batch_size
            acc_sum += acc * batch_size
            num += y.size
        return loss_sum / num, acc_sum / num

    def run_step(self,epoch,train_iter,valid_iter,test_iter, epoch_per_print):
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=0,0,None,None,0,0
        loss_train,acc_train = self.train_loss_acc(train_iter)
        if epoch % epoch_per_print == 0:
            self.run_learning_rate_decay(acc_train)
            loss_test,acc_test = self.evaluate_loss_acc(test_iter)

        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test

    def tile_tensor_firstdim(self,tensor_a, tensor_b):
        '''扩张tensor_a的第一维,使得其和tesnor_b的第一维相同'''
        tile_tensor_a = tf.py_func(self._tile_tensor, [tensor_a, tensor_b], tf.float32)
        return tile_tensor_a

    def _tile_tensor(self,tensor_a, tensor_b):
        #shape = tensor_a.shape
        dim_a = len(tensor_a.shape)
        new_shape = np.ones(dim_a,dtype='int')
        new_shape[0] = tensor_b.shape[0]
        tile_a = np.tile(tensor_a,new_shape)
        #if dim_a > 2:
        #    tile_a = np.tile(tensor_a, (tensor_b.shape[0], *tensor_a.shape[1:-1]),tensor_a.shape[-1])
        #elif dim_a > 1:
        #    tile_a = np.tile(tensor_a, (tensor_b.shape[0], tensor_a.shape[-1]))
        #else:
        #    tile_a = np.tile(tensor_a, (tensor_b.shape[0]))
        return tile_a

    def get_input_shape(self):
        pass

    def initialize(self,ckpt_used):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        #tf.gfile.DeleteRecursively(self.logging_directory)
        #used after create_model
        #self.saver = tf.train.Saver(tf.all_variables(),max_to_keep=self.max_to_keep)
        #self.writer = tf.summary.FileWriter(self.logging_directory, tf.get_default_graph(),
        #                                    max_queue=self.max_queue)
        self.write = tf.summary.create_file_writer(os.path.join(self.logging_directory,self.log_savefile))
        #self.init_op = tf.global_variables_initializer()

        if 'pretrained_model' in self.gConfig:
            self.saver.restore(self.session, self.gConfig['pretrained_model'])
        ckpt = tf.train.get_checkpoint_state(self.gConfig['working_directory'])
        if ckpt and ckpt.model_checkpoint_path and ckpt_used:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            #self.saver.restore(self.session, ckpt.model_checkpoint_path)
            self.net.load_models(self.model_savefile)
            tf.train.Checkpoint(x=self.global_step).restore(tf.train.latest_checkpoint(self.working_directory))
            #graph = tf.get_default_graph()
        else:
            print("Created model with fresh parameters.")
            #self.session.run(self.init_op)
            #graph = tf.get_default_graph()
            self.global_step = tf.constant([0])
            self.net.build(input_shape=self.get_input_shape())
            #self.debug_info(self.net)
            #删除旧的log
        self.net.summary()
        #self.debug_info()


