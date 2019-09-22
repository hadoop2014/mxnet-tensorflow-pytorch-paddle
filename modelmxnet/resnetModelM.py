from mxnet.gluon import loss as gloss,nn
from modelBaseClassM import *

class Residual(nn.HybridBlock):
    def __init__(self,num_channels,use_1x1conv=False,strides=1,**kwargs):
        super(Residual,self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels,kernel_size=3,padding=1,strides=strides)
        self.conv2 = nn.Conv2D(num_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels,kernel_size=1,strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X,**kwargs):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

class resnetModel(modelBaseM):
    def __init__(self,gConfig,getdataClass):
        super(resnetModel,self).__init__(gConfig)
        self.loss = gloss.SoftmaxCrossEntropyLoss()
        self.resizedshape = getdataClass.resizedshape
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(),self.optimizer,
                                     {'learning_rate':self.learning_rate})
        self.input_shape = (self.batch_size,*self.resizedshape)

    def resnet_block(self,num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    def get_net(self):
        residual_arch = self.gConfig['residual_arch']
        ratio = self.gConfig['ratio']
        small_residual_arch = [(pair[0], pair[1] // ratio) for pair in residual_arch]
        conv1_channels = self.gConfig['conv1_channels'] // ratio
        conv1_kernel_size = self.gConfig['conv1_kernel_size']
        conv1_strides = self.gConfig['conv1_strides']
        conv1_padding = self.gConfig['conv1_padding']
        pool1_size = self.gConfig['pool1_size']
        pool1_strides = self.gConfig['pool1_strides']
        pool1_padding = self.gConfig['pool1_padding']
        dense1_hiddens = self.gConfig['dense1_hiddens']
        activation = self.gConfig['activation']
        # 卷积层部分
        self.net.add(nn.Conv2D(conv1_channels, kernel_size=conv1_kernel_size, strides=conv1_strides, padding=conv1_padding),
                nn.BatchNorm(),
                nn.Activation(self.get_activation(activation)),
                nn.MaxPool2D(pool_size=pool1_size, strides=pool1_strides, padding=pool1_padding))
        first_block = True
        for (num_residuls,num_channels) in small_residual_arch:
            self.net.add(self.resnet_block(num_channels,num_residuls,first_block))
            first_block = False
        # 全连接层部分
        self.net.add(nn.GlobalAvgPool2D(),
                     nn.Dense(dense1_hiddens))

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
        acc = (y_hat.argmax(axis=1) == y).sum()
        loss = self.loss(y_hat, y).sum()
        return loss, acc

    def get_input_shape(self):
        return self.input_shape

def create_model(gConfig,ckpt_used,getdataClass):
    model=resnetModel(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model