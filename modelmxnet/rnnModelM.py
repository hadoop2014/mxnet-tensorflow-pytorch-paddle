from mxnet.gluon import loss as gloss,nn,rnn
from mxnet import gluon,init
from modelBaseClassM import *
import math

'''class Rnn(nn.HybridBlock):
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
'''
class RNN(nn.HybridBlock):
    def __init__(self,rnn_hiddens,activation,vocab_size,
                 weight_initializer,bias_initializer,**kwargs):
        super(RNN,self).__init__(**kwargs)
        self.rnn = rnn.RNN(hidden_size=rnn_hiddens,activation=activation,
                                 i2h_weight_initializer=weight_initializer,h2h_weight_initializer=weight_initializer,
                                 i2h_bias_initializer=bias_initializer,h2h_bias_initializer=bias_initializer)
        self.vocab_size = vocab_size
        self.hidden_size = rnn_hiddens
        self.dense = nn.Dense(self.vocab_size)

    def hybrid_forward(self, F, x, *args, **kwargs):
        X = x
        state = args[0]
        Y, state = self.rnn(X,state)
        # 全连接层会首先将Y的形状变成(time_steps * batch_size, num_hiddens)，它的输出
        # 形状为(time_steps * batch_size, vocab_size)
        # output = self.dense(Y.reshape((-1, Y.shape[-1])))
        output = self.dense(F.reshape(Y, (-1, self.hidden_size)))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args,**kwargs)

class rnnModel(modelBaseM):
    def __init__(self,gConfig,getdataClass):
        super(rnnModel,self).__init__(gConfig)
        self.loss = gloss.SoftmaxCrossEntropyLoss()
        self.resizedshape = getdataClass.resizedshape
        self.vocab_size = getdataClass.vocab_size
        self.idx_to_char = getdataClass.idx_to_char
        self.char_to_idx = getdataClass.char_to_idx
        self.clip_gradient = self.gConfig['clip_gradient']
        self.prefixes = self.gConfig['prefixes']
        self.predict_length = self.gConfig['predict_length']
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(),self.optimizer,
                                     {'learning_rate':self.learning_rate,'clip_gradient':self.clip_gradient})
        self.input_shape = (self.resizedshape[0],self.batch_size,self.resizedshape[1])
        #self.input_shape = (self.batch_size,*self.resizedshape)


    def get_net(self):
        input_dim = self.gConfig['input_dim']#1
        input_dim = self.resizedshape[0]
        self.rnn_hiddens =self.gConfig['rnn_hiddens'] #256
        output_dim = self.gConfig['output_dim']#1
        output_dim = self.resizedshape[0]
        activation = self.gConfig['activation']
        weight_initializer = self.get_initializer(self.initializer)
        bias_initializer = self.get_initializer('constant')
        #self.net.add(Rnn(input_dim,self.rnn_hiddens,output_dim,self.batch_size,self.ctx,
        #                 weight_initializer,bias_initializer))
        #self.net.add(RNN(self.rnn_hiddens, activation, self.vocab_size,
        #        weight_initializer, bias_initializer))
        self.net = RNN(self.rnn_hiddens,activation,self.vocab_size,weight_initializer,bias_initializer)

    def init_state(self):
        self.state = self.net.begin_state(batch_size=self.batch_size, ctx=self.ctx)

    def run_train_loss_acc(self, X, y):
        for s in self.state:
            s.detach()
        with autograd.record():
            y_hat,self.state = self.net(X,self.state) #self.state在父类中通过init_state来初始化
            y = y.T.reshape((-1,))
            loss = self.loss(y_hat, y).sum()
        loss.backward()
        if self.global_step == 0:
            self.debug_info()
        self.trainer.step(self.batch_size)
        #self.trainer.step(batch_size=1) #在loss采用mean后，batch_size相应的改成１
        loss = loss.asscalar()
        y = y.astype('float32')
        acc = (y_hat.argmax(axis=1) == y).sum().asscalar()
        #acc = self.evaluate_perplexity(loss)   #必须使用loss.mean的值
        return loss, acc

    def run_eval_loss_acc(self, X, y):
        for s in self.state:
            s.detach()
        y_hat,self.state = self.net(X,self.state)
        y = y.T.reshape((-1,))
        loss = self.loss(y_hat, y).sum().asscalar()
        y = y.astype('float32')
        acc = (y_hat.argmax(axis=1) == y).sum().asscalar()
        #acc = self.evaluate_perplexity(loss)
        return loss, acc

    def evaluate_perplexity(self,loss):
        #rnn中用perplexity取代accuracy
        return math.exp(loss)

    def get_input_shape(self):
        return self.input_shape

    def predict_rnn(self,model):
        for prefix in self.prefixes:
            print(' -', self.predict_rnn_gluon(
                prefix, self.predict_length, model, self.vocab_size, self.ctx, self.idx_to_char,
                self.char_to_idx))

    def predict_rnn_gluon(self,prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                          char_to_idx):
        # 使用model的成员函数来初始化隐藏状态
        state = model.begin_state(batch_size=1, ctx=ctx)
        output = [char_to_idx[prefix[0]]]
        for t in range(num_chars + len(prefix) - 1):
            X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
            X = nd.one_hot(X,vocab_size)
            (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
            if t < len(prefix) - 1:
                output.append(char_to_idx[prefix[t + 1]])
            else:
                output.append(int(Y.argmax(axis=1).asscalar()))
        return ''.join([idx_to_char[i] for i in output])

    def show_net(self,input_shape = None):
        if self.viewIsOn == False:
            return
        #print(self.net)
        title = self.gConfig['taskname']
        input_symbol = mx.symbol.Variable('input_data')
        state_symbol = mx.symbol.Variable('state')
        net,state = self.net(input_symbol,state_symbol)
        mx.viz.plot_network(net, title=title, save_format='png', hide_weights=False,
                                shape=input_shape) \
                .view(directory=self.logging_directory)
        return

    def summary(self):
        self.init_state()
        self.net.summary(nd.zeros(shape=self.get_input_shape(), ctx=self.ctx),
                         self.state)

def create_model(gConfig,ckpt_used,getdataClass):
    model=rnnModel(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model