from modelBaseClassK import *

'''
class lenet(keras.Model):
    def __init__(self,gConfig,activation,classnum,get_padding,**kwargs):
        super(lenet,self).__init__(**kwargs)
        #self.activation =activation # sigmoid
        conv1_channels = gConfig['conv1_channels']  # 6
        conv1_kernel_size = gConfig['conv1_kernel_size']  # 5
        conv1_strides = gConfig['conv1_strides']  # 1
        conv1_padding = gConfig['conv1_padding']  # 0
        pool1_size = gConfig['pool1_size']  # 2
        pool1_strides = gConfig['pool2_size']  # 2
        pool1_padding = gConfig['pool1_padding']  # 0
        conv2_channels = gConfig['conv2_channels']  # 16
        conv2_kernel_size = gConfig['conv2_kernel_size']  # 5
        conv2_strides = gConfig['conv2_striders']  # 1
        conv2_padding = gConfig['conv2_padding']  # 0
        pool2_size = gConfig['pool2_size']  # 2
        pool2_strides = gConfig['pool2_strides']  # 2
        pool2_padding = gConfig['pool2_padding']  # 0
        dense1_hiddens = gConfig['dense1_hiddens']  # 120
        dense2_hiddens = gConfig['dense2_hiddens']  # 84
        dense3_hiddens = gConfig['dense3_hiddens']  # 10
        dense3_hiddens = classnum
        self.conv1 = keras.layers.Conv2D(filters=conv1_channels,kernel_size=conv1_kernel_size,strides=conv1_strides,
                                         padding=get_padding(conv1_padding),activation=activation)
        self.pool1 = keras.layers.MaxPool2D(pool_size=pool1_size,strides = pool1_strides,
                                            padding=get_padding(pool1_padding))
        self.conv2 = keras.layers.Conv2D(filters=conv2_channels,kernel_size=conv2_kernel_size,strides=conv2_strides,
                                         padding=get_padding(conv2_padding),activation=activation)
        self.pool2 = keras.layers.MaxPool2D(pool_size=pool2_size,strides=pool2_strides,
                                            padding=get_padding(pool2_padding))
        self.dense1 = keras.layers.Dense(dense1_hiddens,activation=activation)
        self.dense2 = keras.layers.Dense(dense2_hiddens,activation=activation)
        self.dense3 = keras.layers.Dense(dense3_hiddens)

    def call(self,x):
        x = self.conv1(x)
        #x = self.activation(x)
        x = self.pool1(x)
        x = self.conv2(x)
        #x = self.activation(x)
        x = self.pool2(x)
        pool_out_dim = int(np.prod(x.get_shape()[1:]))
        x = tf.reshape(x,shape=[-1, pool_out_dim])
        x = self.dense1(x)
        #x = self.activation(x)
        x = self.dense2(x)
        #x = self.activation(x)
        x = self.dense3(x)
'''
class lenetModelK(modelBaseK):
    def __init__(self,gConfig,getdataClass):
        super(lenetModelK,self).__init__(gConfig)
        self.resizedshape = getdataClass.resizedshape
        self.classnum = getdataClass.classnum
        self.optimizer = self.get_optimizer(self.optimizer)
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = keras.metrics.SparseCategoricalAccuracy()
        self.get_net()
        self.input_shape = (self.batch_size,self.resizedshape[1],self.resizedshape[2],self.resizedshape[0])

    def get_net(self):
        activation = self.gConfig['activation']#sigmoid
        activation = self.get_activation(activation)
        conv1_channels = self.gConfig['conv1_channels']  # 6
        conv1_kernel_size = self.gConfig['conv1_kernel_size']  # 5
        conv1_strides = self.gConfig['conv1_strides']  # 1
        conv1_padding = self.gConfig['conv1_padding']  # 0
        pool1_size = self.gConfig['pool1_size']  # 2
        pool1_strides = self.gConfig['pool2_size']  # 2
        pool1_padding = self.gConfig['pool1_padding']  # 0
        conv2_channels = self.gConfig['conv2_channels']  # 16
        conv2_kernel_size = self.gConfig['conv2_kernel_size']  # 5
        conv2_strides = self.gConfig['conv2_striders']  # 1
        conv2_padding = self.gConfig['conv2_padding']  # 0
        pool2_size = self.gConfig['pool2_size']  # 2
        pool2_strides = self.gConfig['pool2_strides']  # 2
        pool2_padding = self.gConfig['pool2_padding']  # 0
        dense1_hiddens = self.gConfig['dense1_hiddens']  # 120
        dense2_hiddens = self.gConfig['dense2_hiddens']  # 84
        dense3_hiddens = self.gConfig['dense3_hiddens']  # 10
        dense3_hiddens = self.classnum
        input_channels, input_dim_x, input_dim_y = self.resizedshape
        self.net.add(keras.layers.Conv2D(filters=conv1_channels,kernel_size=conv1_kernel_size,strides=conv1_strides,
                                         padding=self.get_padding()(conv1_padding),activation=activation))
        self.net.add(keras.layers.MaxPool2D(pool_size=pool1_size, strides=pool1_strides,
                                   padding=self.get_padding()(pool1_padding)))
        self.net.add(keras.layers.Conv2D(filters=conv2_channels,kernel_size=conv2_kernel_size,strides=conv2_strides,
                                         padding=self.get_padding()(conv2_padding),activation=activation))
        self.net.add(keras.layers.MaxPool2D(pool_size=pool2_size,strides=pool2_strides,
                                            padding=self.get_padding()(pool2_padding)))
        self.net.add(keras.layers.Flatten(name='Flatten'))
        self.net.add(keras.layers.Dense(dense1_hiddens,activation=activation))
        self.net.add(keras.layers.Dense(dense2_hiddens,activation=activation))
        self.net.add(keras.layers.Dense(dense3_hiddens))
        #self.net = lenet(self.gConfig, activation, self.classnum, self.get_padding())
        self.net.compile(optimizer=self.optimizer,  # Optimizer
                      # Loss function to minimize
                      loss=self.loss,
                      # List of metrics to monitor
                      metrics=self.metrics)


    def run_train_loss_acc(self,X,y,keeps):
        with tf.GradientTape() as tape:
            y_hat = self.net(X,training=True)
            loss = self.loss(y, y_hat)
        grads = tape.gradient(loss, self.net.trainable_variables)
        if self.global_step == 0 or self.global_step == 1:
            self.debug_info(self.net,grads)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
        self.metrics.update_state(y,y_hat)
        acc = self.metrics.result()
        return loss.numpy(),acc.numpy()

    def run_eval_loss_acc(self, X, y, keeps=1.0):
        #loss,acc = self.net.evaluate(X,y)
        y_hat = self.net(X,training=False)
        loss = self.loss(y,y_hat)
        self.metrics.update_state(y,y_hat)
        acc = self.metrics.result()
        return loss.numpy(),acc.numpy()

    def get_input_shape(self):
        return self.input_shape

def create_model(gConfig,ckpt_used,getdataClass):
    model=lenetModelK(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model


