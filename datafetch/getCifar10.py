from  datafetch.getBaseClass import *
import paddle
import numpy as np
import cv2
from PIL import Image

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class getCifar10Data(getdataBase):
    def __init__(self,gConfig):
        super(getCifar10Data,self).__init__(gConfig)
        self.data_path = self.gConfig['data_directory']
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        self.load_data(resize=self.resize,root=self.data_path)

    def load_data(self,resize,root):
        root = os.path.expanduser(root)
        self.transformers = [self.fn_reshape]
        if resize is not None and resize != 0:
            self.transformers = [self.fn_resize]
            self.resizedshape = [self.rawshape[0],self.resize,self.resize]
        self.train_data = paddle.dataset.cifar.train10()#gdata.vision.FashionMNIST(root=root,train=True)
        self.test_data = paddle.dataset.cifar.test10()#gdata.vision.FashionMNIST(root=root,train=False)

    def fn_reshape(self,x):
        return np.reshape(x,self.rawshape)

    def fn_resize(self,x):
        x = np.reshape(x,self.rawshape)
        x = np.transpose(x,axes=[1,2,0])
        if isinstance(self.resize, numeric_types):
            h, w,_ = x.shape
            if h > w:
                wsize = self.resize
                hsize = int(h * wsize / w)
            else:
                hsize = self.resize
                wsize = int(w * hsize / h)
            dsize = (hsize,wsize)
            x = cv2.resize(x,dsize)
            x = np.transpose(x,axes=[2,0,1])
            return x
        else:
            return x

    def transform(self,reader,transformers):
        """
        Create a batched reader.

        :param reader: the data reader to read from.
        :type reader: callable
        :param transformer: a list of transformer.
        :type transformer: list
        :return: the transformed reader.
        :rtype: callable
        """
        def transform_reader():
            for data in reader():
                (X,y) = zip(*data)
                for transformer in transformers:
                    X = np.apply_along_axis(transformer,axis=1,arr=X)
                yield (X,y)

        return transform_reader

    @getdataBase.getdataForUnitest
    def getTrainData(self,batch_size):
        buf_size = self.batch_size * self.cpu_num
        train_iter = paddle.batch(paddle.reader.shuffle(self.train_data, buf_size=buf_size),
                                  batch_size=self.batch_size)
        self.train_iter = self.transform(train_iter,self.transformers)
        return self.train_iter()

    @getdataBase.getdataForUnitest
    def getTestData(self,batch_size):
        buf_size = self.batch_size * self.cpu_num
        test_iter = paddle.batch(paddle.reader.shuffle(self.test_data, buf_size=buf_size),
                                      batch_size=self.batch_size)
        self.test_iter = self.transform(test_iter,self.transformers)
        return self.test_iter()



def create_model(gConfig):
    getdataClass=getCifar10Data(gConfig=gConfig)

    return getdataClass
