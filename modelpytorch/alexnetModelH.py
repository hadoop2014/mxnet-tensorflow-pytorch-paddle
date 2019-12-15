from modelBaseClassH import *
import modelBaseClassH
from torch import nn
import torch.nn.functional as F

class alexnet(nn.Module):
    def __init__(self,gConfig,input_channels,activation,input_dim_x,input_dim_y,classnum,compute_dim_xy):
        super(alexnet,self).__init__()
        self.gConfig = gConfig
        conv1_channels = self.gConfig['conv1_channels']  # 96
        conv1_kernel_size = self.gConfig['conv1_kernel_size']  # 11
        conv1_strides = self.gConfig['conv1_strides']  # 4
        conv1_padding = self.gConfig['conv1_padding']  # 1
        pool1_size = self.gConfig['pool1_size']  # 3
        pool1_strides = self.gConfig['pool1_strides']  # 2
        pool1_padding = self.gConfig['pool1_padding']  # 0
        conv2_channels = self.gConfig['conv2_channels']  # 256
        conv2_kernel_size = self.gConfig['conv2_kernel_size']  # 5
        conv2_strides = self.gConfig['conv2_strides']  # 1
        conv2_padding = self.gConfig['conv2_padding']  # 2
        pool2_size = self.gConfig['pool2_size']  # 3
        pool2_strides = self.gConfig['pool2_strides']  # 2
        pool2_padding = self.gConfig['pool2_padding']  # 0
        conv3_channels = self.gConfig['conv3_channels']  # 384
        conv3_kernel_size = self.gConfig['conv3_kernel_size']  # 3
        conv3_strides = self.gConfig['conv3_strides']  # 1
        conv3_padding = self.gConfig['conv3_padding']  # 1
        conv4_channels = self.gConfig['conv4_channels']  # 384
        conv4_kernel_size = self.gConfig['conv4_kernel_size']  # 3
        conv4_strides = self.gConfig['conv4_strides']  # 1
        conv4_padding = self.gConfig['conv4_padding']  # 1
        conv5_channels = self.gConfig['conv5_channels']  # 256
        conv5_kernel_size = self.gConfig['conv5_kernel_size']  # 3
        conv5_strides = self.gConfig['conv5_strides']  # 1
        conv5_padding = self.gConfig['conv5_padding']  # 1
        pool3_size = self.gConfig['pool3_size']  # 3
        pool3_strides = self.gConfig['pool3_strides']  # 2
        pool3_padding = self.gConfig['pool3_padding']  # 0
        dense1_hiddens = self.gConfig['dense1_hiddens']  # 4096
        drop1_rate = self.gConfig['drop1_rate']  # 0.5
        dense2_hiddens = self.gConfig['dense2_hiddens']  # 4096
        drop2_rate = self.gConfig['drop2_rate']  # 0.5
        dense3_hiddens = self.gConfig['dense3_hiddens']  # 10
        class_num = self.gConfig['class_num']  # 10

        #self.activation =activation # sigmoid
        #self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=conv1_channels,
        #                       kernel_size=conv1_kernel_size,stride=conv1_strides,
        #                       padding=conv1_padding)
        out_dim_x,out_dim_y = compute_dim_xy(input_dim_x,input_dim_y,conv1_kernel_size,conv1_strides,conv1_padding)
        #self.pool1 = nn.MaxPool2d(kernel_size=pool1_size,stride=pool1_strides,
        #                 padding=pool1_padding)
        out_dim_x,out_dim_y = compute_dim_xy(out_dim_x,out_dim_y,pool1_size,pool1_strides,pool1_padding)
        #self.conv2 = nn.Conv2d(in_channels=conv1_channels,out_channels=conv2_channels,
        #                       kernel_size=conv2_kernel_size,stride=conv2_strides,
        #                       padding=conv2_padding)
        out_dim_x,out_dim_y = compute_dim_xy(out_dim_x,out_dim_y,conv2_kernel_size,conv2_strides,conv2_padding)
        #self.pool2 = nn.MaxPool2d(kernel_size=pool2_size,stride=pool2_strides,
        #                 padding=pool2_padding)
        out_dim_x,out_dim_y = compute_dim_xy(out_dim_x,out_dim_y,pool2_size,pool2_strides,pool2_padding)
        #self.conv3 = nn.Conv2d(in_channels=conv2_channels,out_channels=conv3_channels,
        #                       kernel_size=conv3_kernel_size,stride=conv3_strides,
        #                       padding=conv3_padding)
        out_dim_x,out_dim_y = compute_dim_xy(out_dim_x,out_dim_y,conv3_kernel_size,conv3_strides,conv3_padding)
        #self.conv4 = nn.Conv2d(in_channels=conv3_channels, out_channels=conv4_channels,
        #                       kernel_size=conv4_kernel_size, stride=conv4_strides,
        #                       padding=conv4_padding)
        out_dim_x, out_dim_y = compute_dim_xy(out_dim_x, out_dim_y, conv4_kernel_size, conv4_strides, conv4_padding)
        #self.conv5 = nn.Conv2d(in_channels=conv4_channels, out_channels=conv5_channels,
        #                       kernel_size=conv5_kernel_size, stride=conv5_strides,
        #                       padding=conv5_padding)
        out_dim_x, out_dim_y = compute_dim_xy(out_dim_x, out_dim_y, conv5_kernel_size, conv5_strides, conv5_padding)
        #self.pool3 = nn.MaxPool2d( kernel_size=pool3_size, stride=pool3_strides,
        #                     padding=pool3_padding)
        out_dim_x, out_dim_y = compute_dim_xy(out_dim_x, out_dim_y, pool3_size, pool3_strides, pool3_padding)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=conv1_channels,
                      kernel_size=conv1_kernel_size, stride=conv1_strides,
                      padding=conv1_padding),
            activation,
            nn.MaxPool2d(kernel_size=pool1_size, stride=pool1_strides,
                         padding=pool1_padding),
            nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels,
                      kernel_size=conv2_kernel_size, stride=conv2_strides,
                      padding=conv2_padding),
            activation,
            nn.MaxPool2d(kernel_size=pool2_size, stride=pool2_strides,
                         padding=pool2_padding),
            nn.Conv2d(in_channels=conv2_channels,out_channels=conv3_channels,
                               kernel_size=conv3_kernel_size,stride=conv3_strides,
                               padding=conv3_padding),
            activation,
            nn.Conv2d(in_channels=conv3_channels, out_channels=conv4_channels,
                      kernel_size=conv4_kernel_size, stride=conv4_strides,
                      padding=conv4_padding),
            activation,
            nn.Conv2d(in_channels=conv4_channels, out_channels=conv5_channels,
                      kernel_size=conv5_kernel_size, stride=conv5_strides,
                      padding=conv5_padding),
            activation,
            nn.MaxPool2d(kernel_size=pool3_size, stride=pool3_strides,
                         padding=pool3_padding)
        )
        out_dim_x, out_dim_y = compute_dim_xy(input_dim_x, input_dim_y, conv1_kernel_size, conv1_strides, conv1_padding)
        out_dim_x, out_dim_y = compute_dim_xy(out_dim_x, out_dim_y, pool1_size, pool1_strides, pool1_padding)
        out_dim_x, out_dim_y = compute_dim_xy(out_dim_x, out_dim_y, conv2_kernel_size, conv2_strides, conv2_padding)
        out_dim_x, out_dim_y = compute_dim_xy(out_dim_x, out_dim_y, pool2_size, pool2_strides, pool2_padding)
        out_dim_x, out_dim_y = compute_dim_xy(out_dim_x, out_dim_y, conv3_kernel_size, conv3_strides, conv3_padding)
        out_dim_x, out_dim_y = compute_dim_xy(out_dim_x, out_dim_y, conv4_kernel_size, conv4_strides, conv4_padding)
        out_dim_x, out_dim_y = compute_dim_xy(out_dim_x, out_dim_y, conv5_kernel_size, conv5_strides, conv5_padding)
        out_dim_x, out_dim_y = compute_dim_xy(out_dim_x, out_dim_y, pool3_size, pool3_strides, pool3_padding)
        in_features = int(out_dim_x*out_dim_y*conv2_channels)
        #self.dense1 = nn.Linear(in_features=in_features,out_features=dense1_hiddens)
        #self.drop1 = nn.Dropout(drop1_rate)
        #self.dense2 = nn.Linear(in_features=dense1_hiddens,out_features=dense2_hiddens)
        #self.drop2 = nn.Dropout(drop2_rate)
        #self.dense3 = nn.Linear(in_features=dense2_hiddens,out_features=dense3_hiddens)
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features,out_features=dense1_hiddens),
            activation,
            nn.Dropout(drop1_rate),
            nn.Linear(in_features=dense1_hiddens,out_features=dense2_hiddens),
            activation,
            nn.Dropout(drop2_rate),
            nn.Linear(in_features=dense2_hiddens,out_features=dense3_hiddens)
        )

    def forward(self, x):
        features = self.conv(x)
        #pool_out_dim = int(np.prod(x.size()[1:]))
        #x = x.view(-1,pool_out_dim)
        x = self.fc(features.view(x.shape[0],-1))
        return x

class alexnetModel(modelBaseH):
    def __init__(self,gConfig,getdataClass):
        super(alexnetModel,self).__init__(gConfig)
        self.loss = nn.CrossEntropyLoss().to(self.ctx)
        self.resizedshape = getdataClass.resizedshape
        self.classnum = getdataClass.classnum
        self.get_net()
        self.optimizer = self.get_optimizer(self.gConfig['optimizer'],self.net.parameters())
        self.input_shape = (self.batch_size,*self.resizedshape)

    def get_net(self):
        activation = self.gConfig['activation']#sigmoid
        activation = self.get_activation(activation)
        input_channels, input_dim_x, input_dim_y = self.resizedshape
        self.net = alexnet(self.gConfig,input_channels,activation,input_dim_x,input_dim_y,self.classnum,
                         modelBaseH.compute_dim_xy)

    def run_train_loss_acc(self,X,y):
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        loss = self.loss(y_hat, y).sum()
        loss.backward()
        #if self.global_step == 0 or self.global_step == 1:
        #    self.debug_info()
        self.optimizer.step()
        loss = loss.item()
        acc= (y_hat.argmax(dim=1) == y).sum().item()
        return loss,acc

    def run_eval_loss_acc(self, X, y):
        with torch.no_grad():
            #解决GPU　out memory问题
            y_hat = self.net(X)
        acc  = (y_hat.argmax(dim=1) == y).sum().item()
        loss = self.loss(y_hat, y).sum().item()
        return loss,acc

    def get_input_shape(self):
        return self.input_shape


def create_model(gConfig,ckpt_used,getdataClass):
    #用cnnModel实例化一个对象model
    model=alexnetModel(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model