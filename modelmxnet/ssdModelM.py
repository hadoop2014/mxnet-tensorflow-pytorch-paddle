#single shot multibox detection,用于目标检测
#author by wu.wenhua at 2019/10/20
from mxnet.gluon import loss as gloss
from mxnet.ndarray import NDArray
from mxnet import contrib,image,autograd,metric
from datafetch import  commFunction
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data import batchify
from modelBaseClassM import *
from mxnet.gluon.data.dataset import SimpleDataset
import gluoncv

class TinySSD(nn.HybridBlock):
    def __init__(self,num_blocks,num_classes,num_channels,anchor_sizes,anchor_ratios,activation,
                 basenet_filters,img_size,**kwargs):
        super(TinySSD,self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios
        self.activation = activation
        self.basenet_filters = basenet_filters
        self.img_size = img_size
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
        anchors = [None] * self.num_blocks
        class_predictors = [None] * self.num_blocks
        bbox_predictors = [None] * self.num_blocks
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

        return (self.concat_predictors(class_predictors,F).reshape(shape=(0,-1,self.num_classes + 1)),
                self.concat_predictors(bbox_predictors,F).reshape((0,-1,4)),
                F.concat(*anchors,dim=1)*self.img_size)

    def concat_predictors(self,preds,F):
        return F.concat(*[self.flatten_pred(p) for p in preds], dim=1)

    def flatten_pred(self,pred):
        return pred.transpose((0, 2, 3, 1)).flatten()

    def block_forward(self,X,block,size,ratio,class_predictor,bbox_predictor,F):
        Y = block(X)
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
        #self.loss = gloss.SoftmaxCrossEntropyLoss()
        self.loss = gluoncv.loss.SSDMultiBoxLoss()
        self.bbox_loss = gloss.L1Loss()
        self.class_metric = metric.Loss('CrossEntropy')
        self.bbox_metric = metric.Loss('SmoothL1')
        self.resizedshape = getdataClass.resizedshape
        self.classes = getdataClass.classes
        self.classnum = getdataClass.classnum
        self.anchors = None #用于ssd的pretrain模式
        self.ssd_image_size = self.resizedshape[-1] #获取训练数据集的图片大小
        self.activation = self.get_activation(self.gConfig['activation'])
        self.predict_images = self.gConfig['predict_images']
        self.weight_decay = self.gConfig['weight_decay']
        self.predict_threshold = self.gConfig['predict_threshold']
        self.input_shape = (self.batch_size,*self.resizedshape)
        self.test_image_url = self.gConfig['test_image_url']
        self.momentum = self.gConfig['momentum']
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(),self.optimizer,
                                     {'learning_rate':self.learning_rate,'wd':self.weight_decay,'momentum':self.momentum})
        with autograd.train_mode():
            _, _, anchors = self.net(mx.nd.zeros((1, *self.resizedshape),ctx=self.ctx))
        self.ssd_default_train_transform = SSDDefaultTrainTransform(self.ssd_image_size, self.ssd_image_size,
                                                                    anchors.as_in_context(mx.cpu()))

    def get_net(self):
        num_blocks = self.gConfig['num_blocks']
        num_classes = self.classnum
        num_channels = self.gConfig['num_channels']
        anchor_sizes = self.gConfig['anchor_sizes']
        anchor_ratios = self.gConfig['anchor_ratios']
        activation = self.get_activation(self.gConfig['activation'])
        basenet_filters = self.gConfig['basenet_filters']
        self.net = TinySSD(num_blocks,num_classes,num_channels,anchor_sizes,anchor_ratios,activation,basenet_filters,self.ssd_image_size)
    '''
    def run_train_loss_acc(self, X, y):
        with autograd.record():
            cls_preds, bbox_preds, anchors = self.net(X)
            anchors = anchors/self.ssd_image_size
            bbox_labels, bbox_masks, cls_labels = contrib.ndarray.MultiBoxTarget(
                anchors, y, cls_preds.transpose((0, 2, 1)))
            loss = self.calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        if self.global_step == 0 or self.global_step == 1:
            self.debug_info()
        loss.backward()
        self.trainer.step(self.batch_size)
        loss = loss.sum().asscalar()
        n = cls_labels.shape[0]
        self.mae_train = self.bbox_eval(bbox_preds, bbox_labels, bbox_masks)/bbox_labels.size
        acc = (cls_preds.argmax(axis=-1) == cls_labels).mean().asscalar()
        return loss, acc * n
        
    def run_eval_loss_acc(self, X, y):
        cls_preds, bbox_preds,anchors = self.net(X)
        anchors = anchors / self.ssd_image_size   #ssd_image_size默认为１,在调用retrain的ssd模型时，需要归一化为１
        bbox_labels, bbox_masks, cls_labels = contrib.ndarray.MultiBoxTarget(
            anchors, y, cls_preds.transpose((0, 2, 1)))
        n = cls_labels.shape[0]
        acc = (cls_preds.argmax(axis=-1) == cls_labels).mean().asscalar()
        loss=self.calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                       bbox_masks)
        loss = loss.sum().asscalar()
        self.mae_test = self.bbox_eval(bbox_preds, bbox_labels, bbox_masks)/bbox_labels.size
        return loss, acc * n
    '''
    def init_state(self):
        self.bbox_metric.reset()
        self.class_metric.reset()

    def _transform_label(self,label, height=None, width=None):
        """transform label .

            Parameters
            ----------
            label : ndarray.array
                label of the dataset for object detection,  witch shape is [N,5] for example [[id, x, y ,h ,w]]
            height : int
                height of image.
            width : int
                width of image.

            Retrurns
            --------
            label:ndarray.array
                transformed label of the dataset for object detection,  witch shape is [N,5] for example [[x, y ,h ,w, id]]
        """
        label_width = label.shape[1]
        label = np.array(label).ravel()
        min_len = 5
        if len(label) < min_len:
            raise ValueError(
                "Expected label length >= {}, got {}".format(min_len, len(label)))
        gcv_label = label.reshape(-1, label_width)
        # swap columns, gluon-cv requires [xmin-ymin-xmax-ymax-id-extra0-extra1-xxx]
        ids = gcv_label[:, 0].copy()
        gcv_label[:, :4] = gcv_label[:, 1:5]
        gcv_label[:, 4] = ids
        # restore to absolute coordinates
        if height is not None:
            gcv_label[:, (0, 2)] *= width
        if width is not None:
            gcv_label[:, (1, 3)] *= height
        return gcv_label

    def ssd_data_transform(self,X,y):
        batchify_fn = batchify.Tuple(batchify.Stack(), batchify.Stack(), batchify.Stack())
        img = nd.transpose(X, axes=(0, 2, 3, 1)).as_in_context(mx.cpu())
        label = batchify.Stack()([self._transform_label(t, self.ssd_image_size, self.ssd_image_size) for t in y.asnumpy()])
        img, cls_labels, bbox_labels = batchify_fn(list(map(self.ssd_default_train_transform,img,label.asnumpy())))
        return img.as_in_context(self.ctx),cls_labels.as_in_context(self.ctx),bbox_labels.as_in_context(self.ctx)

    def run_train_loss_acc(self, X, y):
        X,cls_labels,bbox_labels=self.ssd_data_transform(X,y)

        with autograd.record():
            cls_preds, bbox_preds, anchors = self.net(X)
            #anchors = anchors/self.ssd_image_size
            #X, cls_labels, bbox_labels =self.ssd_default_train_transform(X,y)

            #dataset = SimpleDataset((X,y))
            #batchify_fn = batchify.Tuple(batchify.Stack(), batchify.Stack(), batchify.Stack())  # stack image, cls_targets, box_targets
            #train_loader = gluon.data.DataLoader(
            #    dataset.transform(SSDDefaultTrainTransform(self.ssd_image_size, self.ssd_image_size, anchors)),
            #    self.batch_size, True, batchify_fn=batchify_fn, last_batch='rollover')
            #X,cls_labels,bbox_labels = train_loader.__iter__()

            #X, cls_labels, bbox_labels = batchify.Stack()(map(self.ssd_default_train_transform,X, y))
            #bbox_labels, bbox_masks, cls_labels = contrib.ndarray.MultiBoxTarget(
            #    anchors, y, cls_preds.transpose((0, 2, 1)))
            #loss = self.calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
            #              bbox_masks)
            loss, cls_loss, box_loss = self.loss(cls_preds, bbox_preds, cls_labels, bbox_labels)
            autograd.backward(loss)
        if self.global_step == 0 or self.global_step == 1:
            self.debug_info()
        #loss.backward()
        # since we have already normalized the loss, we don't want to normalize
        # by batch-size anymore
        self.trainer.step(1)
        #self.trainer.step(self.batch_size)
        loss = loss[0].sum().asscalar()
        n = cls_labels.shape[0]
        self.class_metric.update(0, cls_loss * self.batch_size)
        self.bbox_metric.update(0,box_loss * self.batch_size)
        name_class, loss_class = self.class_metric.get()
        name_bbox, loss_bbox = self.bbox_metric.get()
        #self.mae_train = self.bbox_eval(bbox_preds, bbox_labels, bbox_masks)/bbox_labels.size
        acc = (cls_preds.argmax(axis=-1) == cls_labels).mean().asscalar()
        self.mae_train = loss_bbox
        self.ce_train = loss_class
        return loss, acc * n

    def run_eval_loss_acc(self, X, y):
        X, cls_labels, bbox_labels = self.ssd_data_transform(X, y)
        cls_preds, bbox_preds,anchors = self.net(X)
        #X, cls_labels, bbox_labels = self.ssd_default_train_transform(X, y)
        #anchors = anchors / self.ssd_image_size   #ssd_image_size默认为１,在调用retrain的ssd模型时，需要归一化为１
        #bbox_labels, bbox_masks, cls_labels = contrib.ndarray.MultiBoxTarget(
        #    anchors, y, cls_preds.transpose((0, 2, 1)))
        n = cls_labels.shape[0]
        acc = (cls_preds.argmax(axis=-1) == cls_labels).mean().asscalar()
        #loss=self.calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,bbox_masks)
        loss, cls_loss, box_loss = self.loss(cls_preds, bbox_preds, cls_labels, bbox_labels)
        loss = loss[0].sum().asscalar()
        self.class_metric.update(0, cls_loss * self.batch_size)
        self.bbox_metric.update(0, box_loss * self.batch_size)
        name_class, loss_class = self.class_metric.get()
        name_bbox, loss_bbox = self.bbox_metric.get()
        #self.mae_test = self.bbox_eval(bbox_preds, bbox_labels, bbox_masks)/bbox_labels.size
        self.mae_test = loss_bbox
        self.ce_test = loss_class
        return loss, acc * n

    def calc_loss(self,cls_preds,cls_labels,bbox_preds,bbox_labels,bbox_masks):
        cls = self.loss(cls_preds, cls_labels)
        bbox_preds = bbox_preds.reshape((0,-1))
        bbox = self.bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
        return cls + bbox

    def bbox_eval(self,bbox_preds,bbox_labels,bbox_masks):
        bbox_preds = bbox_preds.reshape((0, -1))
        return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()

    def run_matrix(self, loss_train, loss_test):
        print('global_step %d, mae_train %.6f, ce_train %.6f,mae_test %.6f, ce_loss %.6f' %
              (self.global_step.asscalar(), self.mae_train,self.ce_train,self.mae_test,self.ce_test))
        return self.mae_train,self.mae_test

    def predict_cv(self, model):
        #for image_file in self.predict_images:
        image_file = os.path.join(self.data_directory,self.predict_images[0])
        test_url = self.test_image_url # 'https://raw.githubusercontent.com/zackchase/mxnet-the-straight-dope/master/img/pikachu.jpg'
        gluoncv.utils.download(test_url, image_file)
        #model = gluoncv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=self.classes, pretrained_base=False,ctx=self.ctx)
        #self.model_savefile = '/media/wwyandotte/deeplearning/CodeDepository/GluonCV-Turtoials-master/2.GluonCV_Tutorial/0.2 Object Detection/ssd_512_mobilenet1.0_pikachu.params'
        #model.load_parameters(self.model_savefile,ctx=self.ctx)
        #self.net.load_parameters('/media/wwyandotte/deeplearning/CodeDepository/GluonCV-Turtoials-master/2.GluonCV_Tutorial/0.2 Object Detection/ssd_512_mobilenet1.0_pikachu.params')
        x, image = gluoncv.data.transforms.presets.ssd.load_test(image_file, self.ssd_image_size)#default img_size=512
        cid, score, bbox = model(x.as_in_context(self.ctx))
        commFunction.plt.clf()  # 清楚之前的图片
        ax = gluoncv.utils.viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=self.classes)
        commFunc.plt.show()

    def predict_v1(self, model):
        for image_file in self.predict_images:
            img = image.imread(os.path.join(self.data_directory,image_file))
            feature = image.imresize(img,*self.resizedshape[1:]).astype('float32')
            X = feature.transpose((2, 0, 1)).expand_dims(axis=0)
            cls_preds, bbox_preds,anchors = model(X.as_in_context(self.ctx))
            anchors = anchors / self.ssd_image_size
            cls_probs = cls_preds.softmax().transpose((0, 2, 1))
            bbox_preds = bbox_preds.reshape((0,-1))
            output = contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
            idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
            output = output[0, idx]
            commFunction.set_figsize((5,5))
            self.display(img,output,threshold=self.predict_threshold)

    def display(self,img, output, threshold):
        commFunction.plt.clf() #清楚之前的图片
        fig = commFunction.plt.imshow(img.asnumpy())
        for row in output:
            score = row[1].asscalar()
            if score < threshold:
                continue
            h, w = img.shape[0:2]
            bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
            commFunction.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

    def get_input_shape(self):
        return self.input_shape

    def get_classes(self):
        return self.classes

    def show_net(self,input_shape = None):
        if self.viewIsOn == False:
            return
        if self.gConfig['mode'] == 'pretrain':
            pass
        title = self.gConfig['taskname']
        input_symbol = mx.symbol.Variable('input_data')
        nets = self.net(input_symbol)
        for net in nets:
            mx.viz.plot_network(net, title=title + '.' + str(net), save_format='png', hide_weights=False,
                            shape=input_shape) \
                .view(directory=self.logging_directory,filename='.'.join([self.get_model_name(self.gConfig),net.name,
                                                                          'gv']))
        return

    def transfer_learning(self):
        net = self.get_pretrain_model(ctx=self.ctx,
                                      root=self.working_directory, classes=self.get_classes(),
                                      pretrained_base=False, transfer='voc')
        #net = self.get_pretrain_model(pretrained=True,ctx=self.ctx,root=self.working_directory)
        net.reset_class(self.classes)
        height,width=self.ssd_image_size,self.ssd_image_size
        with autograd.train_mode():
            _, _, anchors = net(mx.nd.zeros((1, 3, height, width),ctx=self.ctx))
        self.ssd_default_train_transform = SSDDefaultTrainTransform(self.ssd_image_size, self.ssd_image_size,
                                                                    anchors.as_in_context(mx.cpu()))
        return net

def create_model(gConfig,ckpt_used,getdataClass):
    model=ssdModel(gConfig=gConfig,getdataClass=getdataClass)
    model.initialize(ckpt_used)
    return model