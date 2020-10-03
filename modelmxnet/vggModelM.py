from mxnet.gluon import loss as gloss,nn
from modelBaseClassM import *


class vggModel(modelBaseM):
    def __init__(self,gConfig,getdataClass):
        super(vggModel,self).__init__(gConfig)
        self.loss = gloss.SoftmaxCrossEntropyLoss()
        self.resizedshape = getdataClass.resizedshape
        self.classnum = getdataClass.classnum
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(),self.optimizer,
                                     {'learning_rate':self.learning_rate})
        self.input_shape = (self.batch_size,*self.resizedshape)

    def vgg_block(self,num_convs,num_channels,activation):
        block = nn.HybridSequential()
        for _ in range(num_convs):
            block.add(nn.Conv2D(num_channels,kernel_size=3,padding=1,activation=self.get_activation(activation),
                                bias_initializer=init.Constant(self.init_bias)))
        block.add(nn.MaxPool2D(pool_size=2,strides=2))
        return block

    def get_net(self):
        conv_arch = self.gConfig['conv_arch']
        ratio = self.gConfig['ratio']
        small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
        dense1_hiddens = self.gConfig['dense1_hiddens'] #// ratio#4096
        dense2_hiddens = self.gConfig['dense2_hiddens'] #// ratio#4096
        dense3_hiddens = self.gConfig['dense3_hiddens'] #10
        dense3_hiddens = self.classnum
        drop1_rate = self.gConfig['drop1_rate'] #0.5
        drop2_rate = self.gConfig['drop2_rate'] #0.5
        activation = self.gConfig['activation'] #relu
        # 卷积层部分
        for (num_convs, num_channels) in small_conv_arch:
            self.net.add(self.vgg_block(num_convs, num_channels,activation=activation))
        # 全连接层部分
        self.net.add(nn.Dense(dense1_hiddens, activation=self.get_activation(activation)),
                     nn.Dropout(drop1_rate),
                     nn.Dense(dense2_hiddens, activation=self.get_activation(activation)),
                     nn.Dropout(drop2_rate),
                     nn.Dense(dense3_hiddens))

    def run_train_loss_acc(self, X, y):
        with autograd.record():
            y_hat = self.net(X)
            loss = self.loss(y_hat, y).sum()
        loss.backward()
        if self.global_step == 0:
            self.debug_info()
        self.trainer.step(self.batch_size)
        loss = loss.asscalar()
        y = y.astype('float32')
        acc = (y_hat.argmax(axis=1) == y).sum().asscalar()
        return loss, acc

    def run_eval_loss_acc(self, X, y):
        y_hat = self.net(X)
        acc = (y_hat.argmax(axis=1) == y).sum().asscalar()
        loss = self.loss(y_hat, y).sum().asscalar()
        return loss, acc

    def get_input_shape(self):
        return self.input_shape

    def get_classes(self):
        return self.classnum

def create_model(gConfig,ckpt_used,getdataClass):
    model=vggModel(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model