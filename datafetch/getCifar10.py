from  datafetch.getBaseClass import *
import numpy as np
import cv2
import sys
from PIL import Image

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class getCifar10DataH(getdataBase):
    def __init__(self,gConfig):
        super(getCifar10DataH,self).__init__(gConfig)
        self.data_path = os.path.join(self.gConfig['data_directory'], 'cifar10')
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        self.load_data(resize=self.resize, root=self.data_path)

    def load_data(self,resize=None,root=""):
        from torch import utils
        from torchvision import datasets,transforms
        root = os.path.expanduser(root)
        transformer = []
        if resize is not None and resize != 0:
            transformer += [transforms.Resize(resize)]
            self.resizedshape = [self.rawshape[0],resize,resize]
        transformer += [transforms.ToTensor()]
        transformer += [transforms.Normalize((0.1307,),(0.3081,))]
        transformer = transforms.Compose(transformer)
        train_data = datasets.CIFAR10(root=root,train=True,download=True,transform=transformer)
        test_data = datasets.CIFAR10(root=root, train=False,download=True,transform=transformer)
        num_workers = 0 if sys.platform.startswith('win32') else self.cpu_num
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.ctx == 'gpu' else {'num_workers':num_workers}
        self.train_iter = utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True,**kwargs)
        self.test_iter = utils.data.DataLoader(test_data,batch_size=self.batch_size,shuffle=True,**kwargs)

class getCifar10DataP(getdataBase):
    def __init__(self,gConfig):
        super(getCifar10DataP,self).__init__(gConfig)
        self.data_path = os.path.join(self.gConfig['data_directory'], 'cifar10')
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        self.load_data(resize=self.resize,root=self.data_path)

    def load_data(self,resize,root):
        import paddle
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
        import paddle
        buf_size = self.batch_size * self.cpu_num
        train_iter = paddle.batch(paddle.reader.shuffle(self.train_data, buf_size=buf_size),
                                  batch_size=self.batch_size)
        self.train_iter = self.transform(train_iter,self.transformers)
        return self.train_iter()

    @getdataBase.getdataForUnitest
    def getTestData(self,batch_size):
        import paddle
        buf_size = self.batch_size * self.cpu_num
        test_iter = paddle.batch(paddle.reader.shuffle(self.test_data, buf_size=buf_size),
                                      batch_size=self.batch_size)
        self.test_iter = self.transform(test_iter,self.transformers)
        return self.test_iter()

class_selector = {
    "mxnet":getCifar10DataP,
    "tensorflow":getCifar10DataP,
    "pytorch":getCifar10DataH,
    "paddle":getCifar10DataP
}

def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
