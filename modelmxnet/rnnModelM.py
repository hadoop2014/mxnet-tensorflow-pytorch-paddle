from mxnet.gluon import loss as gloss,nn
from mxnet import gluon,init
from modelBaseClassM import *

class Rnn(nn.HybridBlock):
    def __init__(self,input_dim,rnn_hiddens,output_dim,batch_size,ctx,
                 weight_initializer,bias_initializer,**kwargs):
        super(Rnn,self).__init__(**kwargs)
        with self.name_scope():
            # 隐藏层参数
            self.W_xh = self.params.get("W_xh",shape=(input_dim,rnn_hiddens),init=weight_initializer)#_one((num_inputs, num_hiddens))
            self.W_hh = self.params.get('W_hh',shape=(rnn_hiddens,rnn_hiddens),init=weight_initializer)#_one((num_hiddens, num_hiddens))
            self.b_h = self.params.get('b_h',shape=(rnn_hiddens),init=bias_initializer)#nd.zeros(num_hiddens, ctx=ctx)
            # 输出层参数
            self.W_hq =self.params.get('W_hq',shape=(rnn_hiddens,output_dim),init=weight_initializer) #_one((num_hiddens, num_outputs))
            self.b_q = self.params.get('b_q',shape=(output_dim),init=bias_initializer)#nd.zeros(num_outputs, ctx=ctx)

            self.initstate = self.params.get('initstate',shape=(batch_size,rnn_hiddens),init=init.Constant(0))

    def hybrid_forward(self, F, X, *args, **kwargs):
        W_xh = kwargs['W_xh']
        W_hh = kwargs['W_hh']
        b_h = kwargs['b_h']
        W_hq = kwargs['W_hq']
        b_q = kwargs['b_q']
        initstate = kwargs['initstate']
        if len(args) == 0:
            H = initstate
        else:
            H = args[0]
        #H = state
        outputs = []
        for x in X:
            H = F.tanh(F.dot(x, W_xh) + F.dot(H, W_hh) + b_h)
            Y = F.dot(H, W_hq) + b_q
            outputs.append(Y)
        return outputs, H


class rnnModel(modelBaseM):
    def __init__(self,gConfig,getdataClass):
        super(rnnModel,self).__init__(gConfig)
        self.loss = gloss.SoftmaxCrossEntropyLoss()
        self.resizedshape = getdataClass.resizedshape
        self.clip_gradient = self.gConfig['clip_gradient']
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(),self.optimizer,
                                     {'learning_rate':self.learning_rate,'clip_gradient':self.clip_gradient})
        #self.input_shape = (self.batch_size,self.gConfig['input_channels'],
        #                                                 self.gConfig['input_dim_x'],self.gConfig['input_dim_y'])
        self.input_shape = (self.batch_size,self.resizedshape[0])


    def get_net(self):
        input_dim = self.gConfig['input_dim']#1
        input_dim = self.resizedshape[0]
        self.rnn_hiddens =self.gConfig['rnn_hiddens'] #256
        output_dim = self.gConfig['output_dim']#1
        output_dim = self.resizedshape[0]
        activation = self.gConfig['activation']
        weight_initializer = self.get_initializer(self.initializer)
        bias_initializer = self.get_initializer('constant')

        self.net.add(Rnn(input_dim,self.rnn_hiddens,output_dim,self.batch_size,self.ctx,
                         weight_initializer,bias_initializer))


    def init_state(self):
        return nd.zeros(shape=(self.batch_size,self.rnn_hiddens),ctx=self.ctx)

    def run_train_loss_acc(self, X, y):
        state = self.init_state()
        with autograd.record():
            y_hat,state = self.net(X,state)
            loss = self.loss(y_hat, y).sum()
        loss.backward()
        if self.global_step == 0:
            self.debug_info()
        self.trainer.step(self.batch_size)
        loss = loss.asscalar()
        y = y.astype('float32')
        acc = (y_hat.argmax(axis=1) == y).sum().asscalar()
        return loss, acc

    def run_loss_acc(self, X, y):
        y_hat = self.net(X)
        acc = (y_hat.argmax(axis=1) == y).sum()
        loss = self.loss(y_hat, y).sum()
        return loss, acc

    def get_input_shape(self):
        return self.input_shape

def create_model(gConfig,ckpt_used,getdataClass):
    #用cnnModel实例化一个对象model
    model=rnnModel(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model