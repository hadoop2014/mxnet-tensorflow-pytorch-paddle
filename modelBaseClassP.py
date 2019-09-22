import paddle.fluid as fluid
from paddle.fluid import core
from visualdl import  LogWriter
from modelBaseClass import *
import cv2
import numpy as np

#深度学习模型的基类
class modelBaseP(modelBase):
    def __init__(self,gConfig):
        super(modelBaseP,self).__init__(gConfig)
        self.learning_rate_value = self.gConfig['learning_rate']
        self.learning_rate_decay_factor = self.gConfig['learning_rate_decay_factor']
        self.decay_steps = self.gConfig['decay_steps']
        self.checkpoint_filename = self.gConfig['checkpoint_filename']
        self.model_savefile =  self.gConfig['model_savefile']+'.' + self.gConfig['framework']
        self.symbol_savefile = self.gConfig['symbol_savefile']+'.' + self.gConfig['framework']
        self.model_pb = os.path.split(self.model_savefile)[-1]
        self.logging_directory = os.path.join(self.gConfig['logging_directory'],self.gConfig['framework'])
        self.init_sigma = self.gConfig['init_sigma']
        self.init_bias = self.gConfig['init_bias']
        self.max_to_keep = self.gConfig['max_to_keep']
        self.max_queue = self.gConfig['max_queue']
        self.places =self.get_places(self.gConfig['ctx'])
        self.executor =  self.get_executor(self.places)
        self.initializer = self.gConfig['initializer']
        self.optimizer = self.gConfig['optimizer']
        self.keeps =self.gConfig['keeps']
        self.global_step = fluid.layers.create_global_var(shape=[1],value=0,dtype='int64',
                                                          persistable=True,
                                                          name='global_step')
        self.global_step_value = 0

        with fluid.name_scope('learning_rate'):
            self.learning_rate = fluid.layers.create_global_var(shape=[1],value=self.learning_rate_value,
                                                                persistable=True,dtype='float32',
                                                                name='learning_rate')
            learning_rate = fluid.layers.exponential_decay(learning_rate=self.learning_rate_value,
                                                                decay_steps=self.decay_steps,
                                                                decay_rate=self.learning_rate_decay_factor,
                                                                staircase=True)
            fluid.layers.assign(learning_rate,self.learning_rate)

    def get_net(self):
        return

    def get_executor(self,places):
        return fluid.Executor(places)

    def get_places(self,ctx):
        assert ctx in self.gConfig['ctxlist'], 'ctx(%s) is invalid,it must one of %s' % \
                                                               (ctx, self.gConfig['ctxlist'])
        if ctx == 'gpu':
            places = fluid.CUDAPlace(0)
        else:
            places = fluid.CPUPlace()
        return places

    def get_initializer(self, initializer):
        assert initializer in self.gConfig['initializerlist'],'initializer(%s) is invalid,it must one of %s' % \
                                                    (initializer, self.gConfig['initializerlist'])
        if initializer == 'normal':
            return fluid.initializer.NormalInitializer(scale=self.init_sigma)
        elif initializer == 'xavier':
            return fluid.initializer.XavierInitializer()
        elif initializer == 'kaiming':
            #何凯明初始化法
            return fluid.initializer.MSRAInitializer()
        elif initializer == 'constant':
            return fluid.initializer.ConstantInitializer(self.init_bias)
        else:
            return None

    def get_optimizer(self,optimizer):
        assert optimizer in self.gConfig['optimizerlist'], 'optimizer(%s) is invalid,it must one of %s' % \
                                                               (optimizer, self.gConfig['optimizerlist'])
        if optimizer == 'sgd':
            return fluid.optimizer.SGDOptimizer(self.learning_rate)
        elif optimizer == 'adadelta':
            return  fluid.optimizer.AdagradOptimizer(self.learning_rate)
        elif optimizer == 'rmsprop':
            return fluid.optimizer.RMSPropOptimizer(self.learning_rate_value)
        elif optimizer == 'adam':
            return fluid.optimizer.AdamOptimizer(self.learning_rate)
        elif optimizer == 'adagrad':
            return fluid.optimizer.AdagradOptimizer(self.learning_rate)
        return None

    def get_activation(self, activation='relu'):
        assert activation in self.gConfig['activationlist'], 'activation(%s) is invalid,it must one of %s' % \
                                                    (activation, self.gConfig['activationlist'])
        return activation

    def get_context(self):
        return self.places

    def get_learningrate(self):
        return  np.array(fluid.global_scope().find_var('learning_rate').get_tensor())

    def get_globalstep(self):
        return self.global_step_value

    def train(self,model_eval,getdataClass,gConfig,num_epochs):
        for epoch in range(num_epochs):
            self.run_epoch(getdataClass, epoch)
        return self.losses_train,self.acces_train,self.losses_valid,self.acces_valid,\
               self.losses_test,self.acces_test

    def debug_info(self,*kargs):
        if len(kargs) == 0:
            return
        (X,y) = kargs
        if self.debugIsOn == False:
            return
        self.debug(X,y)
        return

    def is_trainable(self,var):
        if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                var.desc.type() == core.VarDesc.VarType.READER or \
                var.desc.type() == core.VarDesc.VarType.RAW or \
                var.stop_gradient == True:
            return False
        try:
            if var.trainable == True:
                return var.persistable
        except:
            return False
        return var.persistable

    def debug(self,X,y):
        trainable_vars = list(filter(self.is_trainable, fluid.default_main_program().list_vars()))
        vars,grads = self.run_gradient(trainable_vars,X,y,self.keeps)
        for trainable_var,var,grad in zip(trainable_vars,vars,grads):
            print('\tdebug:%s'%trainable_var.name,
                  '\tshape=',trainable_var.shape,
                  '\tdata.mean=%.6f'%var.mean(),
                  '\tgrad.mean=%.6f' % grad.mean(),
                  '\tdata.std=%.6f'%var.std(),
                  '\tgrad.std=%.6f'%grad.std())

    def run_gradient(self,grads,X,y,keeps):
        grads_value = None
        vars_value = None
        return vars_value,grads_value

    def run_train_loss_acc(self,X,y,keeps):
        loss,acc = None,None
        return loss,acc

    def run_eval_loss_acc(self, X, y, keeps=1.0):
        loss,acc = None,None
        return loss,acc

    def run_learning_rate_decay(self,acc_train):
        return

    def run_step(self,epoch,train_iter,valid_iter,test_iter, epoch_per_print):
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=0,0,None,None,0,0
        num = 0
        for step, (X, y) in enumerate(train_iter):
            try:
                X = X.asnumpy()
                y = y.asnumpy()
                y = np.array(y, dtype='int64')
            except:
                X = np.array(X)
                y = np.array(y, dtype='int64')
            loss,acc  = self.run_train_loss_acc(X,y,self.keeps)
            batch_size = y.size
            loss_train += loss * batch_size
            acc_train += acc * batch_size
            num += batch_size
            self.visualdl_record(self.global_step_value,acc,loss,X[0])
            self.global_step_value += 1
            fluid.layers.assign(np.array([self.global_step_value],dtype=np.int32),
                                self.global_step)
        if epoch % epoch_per_print == 0:
            loss_train = loss_train / num
            acc_train = acc_train / num
            loss_test,acc_test = self.evaluate_accuracy(test_iter,self.executor)
            self.log_test_acc.add_record(int(epoch/epoch_per_print),acc_test)
            self.log_test_lost.add_record(int(epoch/epoch_per_print),loss_test)
        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test

    def evaluate_accuracy(self,data_iter,session):
        acc_sum = 0
        loss_sum =0
        num = 0
        for X, y in data_iter:
            try:
                X = X.asnumpy()
                y = y.asnumpy()
                y = np.array(y, dtype='int64')
            except:
                X = np.array(X)
                y = np.array(y, dtype='int64')
            loss,acc = self.run_eval_loss_acc(X, y)
            batch_size = y.size
            acc_sum += acc * batch_size
            loss_sum += loss * batch_size
            num += y.size
        return loss_sum / num, acc_sum / num

    def saveCheckpoint(self):
        pathname,filename = os.path.split(self.model_savefile)
        fluid.io.save_persistables(self.executor,pathname,fluid.default_main_program(),filename=filename)

    def getSaveFile(self):
        if self.model_savefile == '':
            self.model_savefile = None
            return None
        if self.model_savefile is not None:
            if os.path.exists(self.model_savefile)== False:
               return None
                #文件不存在
        return self.model_savefile

    def removeSaveFile(self):
        if self.model_savefile is not None:
            filename = os.path.join(os.getcwd() , self.model_savefile)
            if os.path.exists(filename):
                os.remove(filename)
        if self.symbol_savefile is not None:
            filename = os.path.join(os.getcwd(),self.symbol_savefile)
            if os.path.exists(filename):
                os.remove(filename)

    def visualdl_record(self,global_step,acc,loss,input_image):
        self.log_train_acc.add_record(global_step, acc)
        self.log_train_loss.add_record(global_step, loss)
        if global_step < self.gConfig['num_samples']:
            if global_step == 0:
                self.log_image.start_sampling()
            idx = self.log_image.is_sample_taken()
            if idx != -1:
                image = np.transpose(input_image, axes=[1, 2, 0])
                self.log_image.set_sample(self.global_step_value, image.shape, image.flatten())
            self.log_hist.add_record(self.global_step_value,
                                     np.array(fluid.global_scope().find_var('conv1/weight').get_tensor()).flatten())
        elif global_step == self.gConfig['num_samples']:
            self.log_image.finish_sampling()

    def initialize(self,ckpt_used):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        self.writer = LogWriter(self.logging_directory,sync_cycle=self.max_queue)
        with self.writer.mode('train') as logger:
            self.log_train_acc = logger.scalar('acc')
            self.log_train_loss = logger.scalar('loss')
        with self.writer.mode('test') as logger:
            self.log_test_acc = logger.scalar('acc')
            self.log_test_lost = logger.scalar('lost')
        with self.writer.mode("train") as logger:
            self.log_image = logger.image("input",num_samples=self.gConfig['num_samples'])
        with self.writer.mode("param") as logger:
            self.log_hist = logger.histogram("weight", num_buckets=self.gConfig['num_samples'])

        if 'pretrained_model' in self.gConfig:
            pathname,filename = os.path.split(self.gConfig['pretrained_model'])
            fluid.io.load_persistables(self.executor,pathname,main_program=fluid.default_main_program(),
                                       filename=filename)#self.net.load_parameters(self.gConfig['pretrained_model'])

        ckpt = self.getSaveFile()
        if ckpt and ckpt_used:
            pathname, filename = os.path.split(ckpt)
            print("Reading model parameters from %s" % ckpt)
            fluid.io.load_persistables(self.executor,pathname,fluid.default_main_program(),
                                       filename=filename)
            self.global_step_value = np.array(fluid.global_scope().find_var('global_step').get_tensor())
        else:
            print("Created model with fresh parameters.")
            #参数初始化
            self.executor.run(fluid.default_startup_program())
            #self.global_step = nd.array([0], ctx=self.ctx)
            #self.debug_info(self.init_op.dumps())
            #self.debug_info(self.net)
            # model.removeSaveFile()

