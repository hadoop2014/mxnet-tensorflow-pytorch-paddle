from modelBaseClassM import *
from mxnet import gluon,init,nd,autograd
from mxnet.gluon import loss as gloss,nn


class houseModel(modelBaseM):
    def __init__(self,gConfig,getdataClass):
        super(houseModel,self).__init__(gConfig)
        self.weight_decay = self.gConfig['weight_decay']
        self.loss = gloss.L2Loss()
        self.resizedshape = getdataClass.resizedshape
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(),self.optimizer,
                                     {'learning_rate':self.learning_rate,'wd':self.weight_decay})
        #self.input_shape = (self.batch_size, gConfig['input_dim'])
        self.input_shape = (self.batch_size,self.resizedshape[0])

    def get_net(self):
        self.net.add(nn.Dense(1))

    def log_rmse(self, features, labels):
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = nd.clip(self.net(features), 1, float('inf'))
        #clipped_preds = self.net(features)
        #print(clipped_preds,labels)
        rmse = nd.sqrt(2 * self.loss(clipped_preds.log(), labels.log()).mean())
        return rmse.asscalar()

    def run_step(self,epoch,train_iter,valid_iter,test_iter, epoch_per_print):
        features = None  # nd.array([])
        labels = None  # nd.array([])
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=None,None,None,None,None,None
        for step, (X, y) in enumerate(train_iter):
            X = X.as_in_context(self.ctx)
            y = y.as_in_context(self.ctx)
            # 房价预先取log,等同于log_rmse
            y_log = y.log()
            if features is None:
                features = nd.array(X, ctx=self.ctx)
                labels = nd.array(y, ctx=self.ctx)
            else:
                features = nd.concat(features, X, dim=0)
                labels = nd.concat(labels, y, dim=0)
            with autograd.record():
                loss = self.loss(self.net(X), y)
            loss.backward()
            if self.global_step == 0:
                self.debug_info()
            self.trainer.step(self.batch_size)
            self.global_step += nd.array([1],ctx=self.ctx)

        if epoch % epoch_per_print == 0:
            # print(features.shape,labels.shape)
            loss_train = loss.mean().asscalar()
            acc_train = self.log_rmse(features, labels)

            for test_feature,test_label in test_iter:
                # test_feature = nd.array(test_feature,ctx=self.ctx)
                test_feature = test_feature.as_in_context(self.ctx)
                test_label = None
                if test_label is not None:
                    self.losses_test.append(self.log_rmse(test_feature, test_label))

            for valid_features,valid_labels in valid_iter:
                valid_features = valid_features.as_in_context(self.ctx)
                valid_labels = valid_labels.as_in_context(self.ctx)
            loss_valid = self.loss(self.net(valid_features), valid_labels).mean().asscalar()
            acc_valid = self.log_rmse(valid_features, valid_labels)

        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test

    def get_input_shape(self):
        return self.input_shape

def create_model(gConfig,ckpt_used,getdataClass):
    #用cnnModel实例化一个对象model
    #gConfig = gConfig#getConfig.get_config(config_file=config_file)
    model=houseModel(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model