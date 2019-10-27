from mxnet.gluon import nn,loss as gloss
from modelBaseClassM import *

class ninModel(modelBaseM):
    def __init__(self,gConfig,getdataClass):
        super(ninModel,self).__init__(gConfig)
        self.loss = gloss.SoftmaxCrossEntropyLoss()
        self.resizedshape = getdataClass.resizedshape
        self.classnum = getdataClass.classnum
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(),self.optimizer,
                                     {'learning_rate':self.learning_rate})
        self.input_shape = (self.batch_size,*self.resizedshape)

    def nin_block(self,num_channels, kernel_size, strides, padding,activation):
        block = nn.HybridSequential()
        block.add(nn.Conv2D(num_channels,kernel_size,strides,padding,activation=activation),
                  nn.Conv2D(num_channels,kernel_size=1,activation=activation),
                  nn.Conv2D(num_channels,kernel_size=1,activation=activation))
        return block

    def get_net(self):
        conv1_channels = self.gConfig['conv1_channels'] #96
        conv1_kernel_size = self.gConfig['conv1_kernel_size'] #11
        conv1_strides = self.gConfig['conv1_strides'] #4
        conv1_padding = self.gConfig['conv1_padding'] #0
        pool1_size = self.gConfig['pool1_size'] #3
        pool1_strides = self.gConfig['pool1_strides'] #2
        pool1_padding = self.gConfig['pool1_padding'] #0
        conv2_channels = self.gConfig['conv2_channels'] #256
        conv2_kernel_size = self.gConfig['conv2_kernel_size'] #5
        conv2_strides = self.gConfig['conv2_strides'] #1
        conv2_padding = self.gConfig['conv2_padding'] #2
        pool2_size = self.gConfig['pool2_size'] #3
        pool2_strides = self.gConfig['pool2_strides'] #2
        pool2_padding = self.gConfig['pool2_padding'] #0
        conv3_channels = self.gConfig['conv3_channels'] #384
        conv3_kernel_size = self.gConfig['conv3_kernel_size'] #3
        conv3_strides = self.gConfig['conv3_strides'] #1
        conv3_padding = self.gConfig['conv3_padding'] #1
        pool3_size = self.gConfig['pool3_size']  # 3
        pool3_strides = self.gConfig['pool3_strides']  # 2
        pool3_padding = self.gConfig['pool3_padding']  # 0
        conv4_channels = self.gConfig['conv4_channels'] #10
        conv4_kernel_size = self.gConfig['conv4_kernel_size'] #3
        conv4_strides = self.gConfig['conv4_strides'] #1
        conv4_padding = self.gConfig['conv4_padding'] #1
        drop1_rate = self.gConfig['drop1_rate'] #0.5
        activation = self.gConfig['activation'] #relu
        classnum = self.classnum
        self.net.add(self.nin_block(conv1_channels,conv1_kernel_size,conv1_strides,conv1_padding,
                                    self.get_activation(activation)),
                     nn.MaxPool2D(pool1_size,pool1_strides),
                     self.nin_block(conv2_channels,conv2_kernel_size,conv2_strides,conv2_padding,
                                    self.get_activation(activation)),
                     nn.MaxPool2D(pool2_size,pool2_strides),
                     self.nin_block(conv3_channels,conv3_kernel_size,conv3_strides,conv3_padding,
                                    self.get_activation(activation)),
                     nn.MaxPool2D(pool3_size,pool3_strides),
                     nn.Dropout(drop1_rate),
                     self.nin_block(conv4_channels,conv4_kernel_size,conv4_strides,conv4_padding,
                                    self.get_activation(activation)),
                     nn.GlobalAvgPool2D(),
                     nn.Flatten())

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

    def get_classnum(self):
        return self.classnum

def create_model(gConfig,ckpt_used,getdataClass):
    model=ninModel(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model
