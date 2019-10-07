from mxnet.gluon import nn,loss as gloss
from modelBaseClassM import *

class alexnetModel(modelBaseM):
    def __init__(self,gConfig,getdataClass):
        super(alexnetModel,self).__init__(gConfig)
        self.loss = gloss.SoftmaxCrossEntropyLoss()
        self.resizedshape = getdataClass.resizedshape
        self.classnum = getdataClass.classnum
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(),self.optimizer,
                                     {'learning_rate':self.learning_rate})
        self.input_shape = (self.batch_size,*self.resizedshape)

    def get_net(self):
        conv1_channels = self.gConfig['conv1_channels'] #96
        conv1_kernel_size = self.gConfig['conv1_kernel_size'] #11
        conv1_strides = self.gConfig['conv1_striders'] #4
        conv1_padding = self.gConfig['conv1_padding'] #1
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
        conv4_channels = self.gConfig['conv4_channels'] #384
        conv4_kernel_size = self.gConfig['conv4_kernel_size'] #3
        conv4_strides = self.gConfig['conv4_strides'] #1
        conv4_padding = self.gConfig['conv4_padding'] #1
        conv5_channels = self.gConfig['conv5_channels'] #256
        conv5_kernel_size = self.gConfig['conv5_kernel_size'] #3
        conv5_strides = self.gConfig['conv5_strides'] #1
        conv5_padding = self.gConfig['conv5_padding'] #1
        pool3_size = self.gConfig['pool3_size'] #3
        pool3_strides = self.gConfig['pool3_strides'] #2
        pool3_padding = self.gConfig['pool3_padding'] #0
        dense1_hiddens = self.gConfig['dense1_hiddens'] #4096
        drop1_rate = self.gConfig['drop1_rate'] #0.5
        dense2_hiddens = self.gConfig['dense2_hiddens'] #4096
        drop2_rate = self.gConfig['drop2_rate'] #0.5
        dense3_hiddens = self.gConfig['dense3_hiddens'] #10
        dense3_hiddens = self.classnum
        activation = self.gConfig['activation'] #relu
        self.net.add(nn.Conv2D(conv1_channels, kernel_size=conv1_kernel_size, strides=conv1_strides,
                               activation=self.get_activation(activation)),
                     nn.MaxPool2D(pool_size=pool1_size, strides=pool1_strides),
                     # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
                     nn.Conv2D(conv2_channels, kernel_size=conv2_kernel_size, padding=conv2_padding,
                               activation=self.get_activation(activation)),
                     nn.MaxPool2D(pool_size=pool2_size, strides=pool2_strides),
                     # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
                     # 前两个卷积层后不使用池化层来减小输入的高和宽
                     nn.Conv2D(conv3_channels, kernel_size=conv3_kernel_size, padding=conv3_padding,
                               activation=self.get_activation(activation)),
                     nn.Conv2D(conv4_channels, kernel_size=conv4_kernel_size, padding=conv4_padding,
                               activation=self.get_activation(activation)),
                     nn.Conv2D(conv5_channels, kernel_size=conv5_kernel_size, padding=conv5_padding,
                               activation=self.get_activation(activation)),
                     nn.MaxPool2D(pool_size=pool3_size, strides=pool3_strides),
                     # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
                     nn.Dense(dense1_hiddens, activation=self.get_activation(activation)),
                     nn.Dropout(drop1_rate),
                     nn.Dense(dense2_hiddens, activation=self.get_activation(activation)),
                     nn.Dropout(drop2_rate),
                     # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
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

    def get_classnum(self):
        return self.classnum

def create_model(gConfig,ckpt_used,getdataClass):
    model=alexnetModel(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model
