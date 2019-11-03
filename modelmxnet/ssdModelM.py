#single shot multibox detection,用于目标检测
#author by wu.wenhua at 2019/10/20
from mxnet.gluon import loss as gloss,nn,rnn
from mxnet.ndarray import NDArray
from mxnet import contrib
from modelBaseClassM import *

class TinySSD(nn.HybridBlock):
    def __init__(self,num_blocks,num_classes,num_channels,anchor_sizes,anchor_ratios,activation,
                 basenet_filters,**kwargs):
        super(TinySSD,self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios
        self.activation = activation
        self.basenet_filters = basenet_filters
        def class_predictor(num_anchors, num_classes):
            return nn.Conv2D(channels=num_anchors * (num_classes + 1),kernel_size=3,padding=1)
        def bbox_predictor(num_anchors):
            return nn.Conv2D(channels=num_anchors * 4, kernel_size=3, padding=1)
        for i in range(self.num_blocks):
            # 即赋值语句self.blk_i = get_blk(i)
            sizes = self.anchor_sizes[i]
            ratios = self.anchor_ratios[i]
            num_anchors = len(sizes) + len(ratios) - 1
            setattr(self, 'block_%d'%i, self.get_block(i))
            setattr(self,'class_predictor_%d'%i, class_predictor(num_anchors,num_classes))
            setattr(self,'bbox_predictor_%d'%i, bbox_predictor(num_anchors))


    def hybrid_forward(self, F, X, *args, **kwargs):
        anchors = [None] * 5
        class_predictors = [None] * 5
        bbox_predictors = [None] * 5
        for i in range(self.num_blocks):
            sizes = self.anchor_sizes[i]
            ratios = self.anchor_ratios[i]
            if isinstance(X, NDArray):
                F_module = __import__('mxnet.contrib.ndarray',fromlist=['ndarray'])
            else:
                F_module = __import__('mxnet.contrib.symbol',fromlist=['symbol'])
            X, anchors[i], class_predictors[i],bbox_predictors[i] = \
                self.block_forward(X,
                                   getattr(self, 'block_%d'%i),
                                   sizes,ratios,
                                   getattr(self,'class_predictor_%d'%i),
                                   getattr(self,'bbox_predictor_%d'%i),F_module)

        return (F.concat(*anchors,dim=1),
                self.concat_predictors(class_predictors,F).reshape(shape=(0,-1,self.num_classes + 1)),
                self.concat_predictors(bbox_predictors,F))

    def concat_predictors(self,preds,F):
        return F.concat(*[self.flatten_pred(p) for p in preds], dim=1)

    def flatten_pred(self,pred):
        return pred.transpose((0, 2, 3, 1)).flatten()

    def block_forward(self,X,block,size,ratio,class_predictor,bbox_predictor,F):
        Y = block(X)
        #anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=size, ratios=ratio)
        anchors = F.MultiBoxPrior(Y, sizes = size, ratios = ratio)
        class_predictors = class_predictor(Y)
        bbox_predictors = bbox_predictor(Y)
        return (Y, anchors, class_predictors, bbox_predictors)

    def get_block(self,block_index):
        if block_index == 0:
            block = self.base_net()
        elif block_index == self.num_blocks - 1:
            block = nn.GlobalMaxPool2D()
        else:
            block = self.down_sample_block(self.num_channels)
        return block

    def base_net(self):
        block = nn.HybridSequential()
        for num_filters in self.basenet_filters:#[16, 32, 64]
            block.add(self.down_sample_block(num_filters))
        return block

    def down_sample_block(self,num_channels):
        block = nn.HybridSequential()
        for _ in range(2):
            block.add(nn.Conv2D(num_channels,kernel_size=3,padding=1),
                      nn.BatchNorm(in_channels=num_channels),
                      nn.Activation(self.activation))
        block.add(nn.MaxPool2D(pool_size=2))
        return block

class ssdModel(modelBaseM):
    def __init__(self,gConfig,getdataClass):
        super(ssdModel,self).__init__(gConfig)
        self.loss = gloss.SoftmaxCrossEntropyLoss()
        self.bbox_loss = gloss.L1Loss()
        self.resizedshape = getdataClass.resizedshape
        self.activation = self.get_activation(self.gConfig['activation'])
        self.classnum = getdataClass.classnum
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(),self.optimizer,
                                     {'learning_rate':self.learning_rate})
        self.input_shape = (self.batch_size,*self.resizedshape)

    def get_net(self):
        num_blocks = self.gConfig['num_blocks']
        num_classes = self.classnum
        num_channels = self.gConfig['num_channels']
        anchor_sizes = self.gConfig['anchor_sizes']
        anchor_ratios = self.gConfig['anchor_ratios']
        activation = self.get_activation(self.gConfig['activation'])
        basenet_filters = self.gConfig['basenet_filters']
        self.net = TinySSD(num_blocks,num_classes,num_channels,anchor_sizes,anchor_ratios,activation,basenet_filters)

    def run_train_loss_acc(self, X, y):
        with autograd.record():
            #y_hat = self.net(X)
            anchors, cls_preds, bbox_preds = self.net(X)
            bbox_labels, bbox_masks, cls_labels = contrib.ndarray.MultiBoxTarget(
                anchors, y, cls_preds.transpose((0, 2, 1)))
            loss = self.calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
            #loss = self.loss(y_hat, y).sum()
        loss.backward()
        if self.global_step == 0:
            self.debug_info()
        self.trainer.step(self.batch_size)
        loss = loss.sum().asscalar()
        y = y.astype('float32')
        #acc = (y_hat.argmax(axis=1) == y).sum().asscalar()
        acc = (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()
        return loss, acc

    def calc_loss(self,cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks):
        cls = self.loss(cls_preds, cls_labels)
        bbox = self.bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
        return cls + bbox

    def bbox_eval(self,bbox_preds,bbox_labels,bbox_masks):
        return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()

    def run_eval_loss_acc(self, X, y):
        #y_hat = self.net(X)
        anchors, cls_preds, bbox_preds = self.net(X)
        #acc = (y_hat.argmax(axis=1) == y).sum().asscalar()
        bbox_labels, bbox_masks, cls_labels = contrib.ndarray.MultiBoxTarget(
            anchors, y, cls_preds.transpose((0, 2, 1)))
        acc = (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()
        #loss = self.loss(y_hat, y).sum().asscalar()
        loss=self.calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                       bbox_masks)
        loss = loss.sum().asscalar()
        return loss, acc

    def get_input_shape(self):
        return self.input_shape

    def get_classnum(self):
        return self.classnum

    def show_net(self,input_shape = None):
        if self.viewIsOn == False:
            return
        #print(self.net)
        title = self.gConfig['taskname']
        input_symbol = mx.symbol.Variable('input_data')
        nets = self.net(input_symbol)
        for net in nets:
            mx.viz.plot_network(net, title=title + '.' + str(net), save_format='png', hide_weights=False,
                            shape=input_shape) \
                .view(directory=self.logging_directory)
        return

def create_model(gConfig,ckpt_used,getdataClass):
    model=ssdModel(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model