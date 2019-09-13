import numpy as np
from modelBaseClass import  *
from torch import optim,nn
from functools import partial
import torch
from torchvision import models
from tensorboardX import SummaryWriter
from torchsummary import summary

#深度学习模型的基类
class modelBaseH(modelBase):
    def __init__(self,gConfig):
        super(modelBaseH,self).__init__(gConfig)
        self.learning_rate = self.gConfig['learning_rate']
        self.learning_rate_decay_factor = self.gConfig['learning_rate_decay_factor']
        self.model_savefile = self.gConfig['model_savefile']+'.' + self.gConfig['framework']
        self.symbol_savefile = self.gConfig['symbol_savefile']+'.' + self.gConfig['framework']
        self.logging_directory = os.path.join(self.logging_directory,self.gConfig['framework'])
        self.viewIsOn = self.gConfig['viewIsOn'.lower()]
        self.max_to_keep = self.gConfig['max_to_keep']
        self.ctx =self.get_ctx(self.gConfig['ctx'])
        #self.optimizer = self.gConfig['optimizer']
        self.init_sigma = self.gConfig['init_sigma']
        self.init_bias = self.gConfig['init_bias']
        self.momentum = self.gConfig['momentum']
        self.initializer = self.gConfig['initializer']
        self.max_queue = self.gConfig['max_queue']
        #self.init_op = self.get_initializer(self.initializer)
        self.weight_initializer = self.get_initializer(self.initializer)
        self.bias_initializer = self.get_initializer('constant')
        self.global_step = torch.tensor(0,dtype=torch.int64,device=self.ctx)
        self.set_default_tensor_type()  #设置默认的tensor在ｇｐｕ还是在ｃｐｕ上运算
        self.net = nn.Module()

    def get_net(self):
        return

    def get_ctx(self,ctx):
        assert ctx in self.gConfig['ctxlist'], 'ctx(%s) is invalid,it must one of %s' % \
                                                               (ctx, self.gConfig['ctxlist'])
        if ctx == 'gpu':
            ctx = torch.device(type='cuda',index=0) #,index=0)
        else:
            ctx = torch.device(type='cpu')#,index=0)
        return ctx

    def get_initializer(self, initializer):
        assert initializer in self.gConfig['initializerlist'],'initializer(%s) is invalid,it must one of %s' % \
                                                    (initializer, self.gConfig['initializerlist'])
        if initializer == 'normal':
            return partial(nn.init.normal_,std=self.init_sigma)
        elif initializer == 'xavier':
            return partial(nn.init.xavier_uniform_)
        elif initializer == 'kaiming':
            #何凯明初始化法
            return partial(nn.init.kaiming_uniform_,mode='fan_in')
        elif initializer == 'constant':
            return partial(nn.init.constant_,val=self.init_bias)
        else:
            return None

    def params_initialize(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            #print(self.weight_initializer,'\tnow initializer %s'%module)
            self.weight_initializer(module.weight.data)
            self.bias_initializer(module.bias.data)

    def get_optimizer(self,optimizer,parameters):
        assert optimizer in self.gConfig['optimizerlist'], 'optimizer(%s) is invalid,it must one of %s' % \
                                                               (optimizer, self.gConfig['optimizerlist'])
        if optimizer == 'sgd':
            return optim.SGD(params=parameters,lr=self.learning_rate)
        elif optimizer == 'adadelta':
            return optim.Adagrad(params=parameters,lr=self.learning_rate)
        elif optimizer == 'rmsprop':
            return optim.RMSprop(params=parameters,lr=self.learning_rate)
        elif optimizer == 'adam':
            return optim.Adam(params=parameters,lr=self.learning_rate)
        elif optimizer == 'adagrad':
            return optim.Adagrad
        elif optimizer == 'momentum':
            return optim.SGD(params=parameters,lr=self.learning_rate,momentum=self.momentum)
        return None

    def get_activation(self, activation='relu'):
        assert activation in self.gConfig['activationlist'], 'activation(%s) is invalid,it must one of %s' % \
                                                    (activation, self.gConfig['activationlist'])
        if activation == 'sigmoid':
            return torch.sigmoid
        elif activation == 'relu':
            return torch.relu

    def get_context(self):
        return self.ctx

    def get_learningrate(self):
        return self.learning_rate

    def get_globalstep(self):
        return self.global_step.item()

    def get_input_shape(self):
        return tuple()


    def saveCheckpoint(self):
        torch.save(self.net.state_dict(),self.model_savefile)
        torch.save(self.global_step,self.symbol_savefile)


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


    def debug_info(self,info = None):
        if self.debugIsOn == False:
            return
        if info is not None:
            print('debug:%s'%info)
            return
        for layer in self.net.children():
            self.debug(layer)
        print('\n')
        return

    def debug(self,layer,name=''):
        print('\tdebug:%s(%s):weight' % (name,layer._get_name()),
              '\tshape=',layer.weight.shape,
              '\tdata.mean=%.6f'%layer.weight.data.mean().item(),
              '\tgrad.mean=%.6f' % layer.weight.grad.mean().item(),
              '\tdata.std=%.6f'%layer.weight.data.std(),
              '\tgrad.std=%.6f' % layer.weight.grad.std())
        print('\tdebug:%s(%s):bias' % (name, layer._get_name()),
              '\tshape=', layer.bias.shape,
              '\t\tdata.mean=%.6f' % layer.bias.data.mean().item(),
              '\tgrad.mean=%.6f' % layer.bias.grad.mean().item(),
              '\tdata.std=%.6f' % layer.bias.data.std(),
              '\tgrad.std=%.6f' % layer.bias.grad.std())

    def image_record(self,global_step,tag,input_image):
        if global_step < self.gConfig['num_samples']:
            self.writer.add_image(tag,input_image,global_step)


    def run_train_loss_acc(self,X,y):
        loss,acc = None,None
        return loss,acc

    def run_loss_acc(self,X,y):
        loss,acc = None,None
        return loss,acc

    def evaluate_accuracy(self, data_iter):
        acc_sum = 0
        loss_sum = 0
        n = 0
        for X, y in data_iter:
            try:
                X = X.asnumpy()
                y = y.asnumpy()
            except:
                X = np.array(X)
                y = np.array(y)
            X = torch.tensor(X,device=self.ctx)
            y = torch.tensor(y,device=self.ctx,dtype=torch.long)
            loss,acc = self.run_loss_acc(X,y)
            acc_sum += acc
            loss_sum += loss
            n += y.size()[0]
            self.writer.add_scalar('test/loss',loss,self.global_step.item())
            self.writer.add_scalar('test/accuracy',acc,self.global_step.item())
        return loss_sum / n, acc_sum / n

    def run_step(self,epoch,train_iter,valid_iter,test_iter, epoch_per_print):
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=0,0,None,None,0,0
        num = 0
        for step, (X, y) in enumerate(train_iter):
            try:
                X = X.asnumpy()
                y = y.asnumpy()
            except:
                X = np.array(X)
                y = np.array(y)
            self.image_record(self.global_step.item(),'input/image',X[0])
            X = torch.tensor(X,device=self.ctx)
            y = torch.tensor(y,device=self.ctx,dtype=torch.long)
            loss, acc = self.run_train_loss_acc(X, y)
            loss_train += loss
            acc_train += acc
            num += y.size()[0]
            self.writer.add_scalar('train/loss',loss,self.global_step.item())
            self.writer.add_scalar('train/accuracy',acc,self.global_step.item())
            self.global_step += 1
        #nd.waitall()
        if epoch % epoch_per_print == 0:
            # print(features.shape,labels.shape)
            loss_train = loss_train / num
            acc_train = acc_train / num
            loss_test,acc_test = self.evaluate_accuracy(test_iter)
            #self.debug_info()

        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test

    def set_default_tensor_type(self):
        if self.ctx == torch.device(type='cuda',index=0):
            # 如果设置为cuda的类型，则所有操作都在GPU上进行
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)

    def initialize(self,ckpt_used):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        self.clear_logging_directory(self.logging_directory)

        self.writer = SummaryWriter(logdir=self.logging_directory,max_queue=self.max_queue)
        if 'pretrained_model' in self.gConfig:
            self.net.load_parameters(self.gConfig['pretrained_model'])

        ckpt = self.getSaveFile()
        if ckpt and ckpt_used:
            print("Reading model parameters from %s" % ckpt)
            #self.net = torch.load(ckpt,map_location=self.ctx)
            self.net.load_state_dict(torch.load(ckpt))
            self.global_step = torch.load(self.symbol_savefile,map_location=self.ctx)
        else:
            print("Created model with fresh parameters.")
            self.net.apply(self.params_initialize)
            self.global_step = torch.tensor(0,dtype=torch.int64,device=self.ctx)
            self.debug_info(self.net)
            dummy_input = torch.zeros(*self.get_input_shape())
            summary(self.net,input_size=self.get_input_shape()[1:],batch_size=self.get_input_shape()[0])
            #self.writer.add_graph(self.net,dummy_input)
