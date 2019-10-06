from mxnet.gluon import nn
from mxnet import nd, gluon,autograd,init
import mxnet as mx
import numpy as np
from modelBaseClass import  *

#深度学习模型的基类
class modelBaseM(modelBase):
    def __init__(self,gConfig):
        super(modelBaseM,self).__init__(gConfig)
        self.learning_rate = self.gConfig['learning_rate']
        self.learning_rate_decay_factor = self.gConfig['learning_rate_decay_factor']
        self.viewIsOn = self.gConfig['viewIsOn'.lower()]
        self.max_to_keep = self.gConfig['max_to_keep']
        self.ctx =self.get_ctx(self.gConfig['ctx'])
        self.optimizer = self.get_optimizer(self.gConfig['optimizer'])
        self.init_sigma = self.gConfig['init_sigma']
        self.init_bias = self.gConfig['init_bias']
        self.weight_initializer = self.get_initializer(self.gConfig['initializer'])
        self.bias_initializer = self.get_initializer('constant')
        self.global_step = nd.array([0],self.ctx)
        self.state = None #用于rnn,lstm等
        self.net = nn.HybridSequential()

    def get_net(self):
        return

    def get_ctx(self,ctx):
        assert ctx in self.gConfig['ctxlist'], 'ctx(%s) is invalid,it must one of %s' % \
                                                               (ctx, self.gConfig['ctxlist'])
        if ctx == 'gpu':
            ctx = mx.gpu(0)
        else:
            ctx = mx.cpu(0)
        return ctx

    def get_initializer(self, initializer):
        assert initializer in self.gConfig['initializerlist'],'initializer(%s) is invalid,it must one of %s' % \
                                                    (initializer, self.gConfig['initializerlist'])
        if initializer == 'normal':
            return init.Normal(sigma=self.init_sigma)
        elif initializer == 'xavier':
            return init.Xavier()
        elif initializer == 'kaiming':
            #何凯明初始化法
            return init.Xavier(rnd_type='uniform',factor_type='in',magnitude=np.sqrt(2))
        elif initializer == 'constant':
            return init.Constant(self.init_bias)
        else:
            return None

    def get_optimizer(self,optimizer):
        assert optimizer in self.gConfig['optimizerlist'], 'optimizer(%s) is invalid,it must one of %s' % \
                                                               (optimizer, self.gConfig['optimizerlist'])
        return optimizer

    def get_activation(self, activation='relu'):
        assert activation in self.gConfig['activationlist'], 'activation(%s) is invalid,it must one of %s' % \
                                                    (activation, self.gConfig['activationlist'])
        return activation

    def get_context(self):
        return self.ctx

    def get_learningrate(self):
        return self.learning_rate

    def get_globalstep(self):
        return self.global_step.asscalar()

    def get_input_shape(self):
        pass

    def init_state(self):
        pass

    def show_net(self,input_shape = None):
        if self.viewIsOn == False:
            return
        #print(self.net)
        title = self.gConfig['taskname']
        input_symbol = mx.symbol.Variable('input_data')
        net = self.net(input_symbol)
        mx.viz.plot_network(net, title=title, save_format='png', hide_weights=False,
                            shape=input_shape) \
                .view(directory=self.logging_directory)
        return

    def saveCheckpoint(self):
        self.net.save_parameters(self.model_savefile)
        nd.save(self.symbol_savefile,self.global_step)

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

    def train(self,model_eval,getdataClass,gConfig,num_epochs):
        for epoch in range(num_epochs):
            self.run_epoch(getdataClass,epoch)

        return self.losses_train,self.acces_train,self.losses_valid,self.acces_valid,\
               self.losses_test,self.acces_test

    def debug_info(self, info=None):
        if self.debugIsOn == False:
            return
        if info is not None:
            print('debug:%s' % info)
            return
        self.debug(self.net)
        print('\n')
        return

    def debug(self, layer, name=''):
        if len(layer._children) != 0:
            for block in layer._children.values():
                self.debug(block, layer.name)
        if str(layer.name).find('pool') < 0 and \
                str(layer.name).find('dropout') < 0 and \
                str(layer.name).find('batchnorm') and \
                str(layer.name).find('relu') < 0 and \
                str(layer.name).find('sigmoid') < 0:
            for param in layer.params:
                parameter = layer.params[param]
                print('\tdebug:%s(%s)' % (name, param),
                      '\tshape=', parameter.shape,
                      '\tdata.mean=%f' % parameter.data().mean().asscalar(),
                      '\tgrad.mean=%f' % parameter.grad().mean().asscalar(),
                      '\tdata.std=%.6f' % parameter.data().asnumpy().std(),
                      '\tgrad.std=%.6f' % parameter.grad().asnumpy().std())

    def predict_rnn(self, model):
        #仅用于rnn网络的句子预测
        pass

    def run_train_loss_acc(self,X,y):
        loss,acc = None,None
        return loss,acc

    def run_eval_loss_acc(self, X, y):
        loss,acc = None,None
        return loss,acc

    def run_perplexity(self, loss_train, loss_test):
        pass

    def train_loss_acc(self,data_iter):
        acc_sum = 0
        loss_sum = 0
        n = 0
        self.init_state()  # 仅用于RNN,LSTM等
        for X, y in data_iter:
            X = nd.array(X, ctx=self.ctx)
            y = nd.array(y, ctx=self.ctx)
            y = y.astype('float32')
            loss, acc = self.run_train_loss_acc(X, y)
            acc_sum += acc
            loss_sum += loss
            n += y.size
            self.global_step += nd.array([1],ctx=self.ctx)
        return loss_sum / n, acc_sum / n

    def evaluate_loss_acc(self, data_iter):
        acc_sum = 0
        loss_sum = 0
        n = 0
        self.init_state()  #仅用于RNN,LSTM等
        for X, y in data_iter:
            X = nd.array(X,ctx=self.ctx)
            y = nd.array(y,ctx=self.ctx)
            y = y.astype('float32')
            loss,acc = self.run_eval_loss_acc(X, y)
            acc_sum += acc
            loss_sum += loss
            n += y.size
        return loss_sum / n, acc_sum / n

    def run_step(self,epoch,train_iter,valid_iter,test_iter, epoch_per_print):
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=0,0,None,None,0,0
        #nd.waitall()
        loss_train,acc_train = self.train_loss_acc(train_iter)
        loss_test, acc_test = self.evaluate_loss_acc(test_iter)
        if epoch % epoch_per_print == 0:
            #loss_test,acc_test = self.evaluate_loss_acc(test_iter)
            self.run_perplexity(loss_train, loss_test)   #仅用于rnn,lstm等
            self.predict_rnn(self.net)    #仅用于rnn,lstm等
        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test

    def summary(self):
        self.net.summary(nd.zeros(shape=self.get_input_shape(),ctx=self.ctx))

    def hybridize(self):
        self.net.hybridize()

    def initialize(self,ckpt_used):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        self.show_net(input_shape={'input_data':self.get_input_shape()})
        if 'pretrained_model' in self.gConfig:
            self.net.load_parameters(self.gConfig['pretrained_model'])
        ckpt = self.getSaveFile()
        if ckpt and ckpt_used:
            print("Reading model parameters from %s" % ckpt)
            self.net.load_parameters(ckpt, ctx=self.ctx)
            self.global_step = nd.load(self.symbol_savefile)[0]
        else:
            print("Created model with fresh parameters.")
            self.net.initialize(self.weight_initializer, force_reinit=True, ctx=self.ctx)
            self.global_step = nd.array([0], ctx=self.ctx)
            self.debug_info(self.weight_initializer.dumps())
            self.debug_info(self.net)
            # model.removeSaveFile()
            self.summary()
        self.hybridize()