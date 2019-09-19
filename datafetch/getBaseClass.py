import os
import re
import sys

numeric_types = (float, int)

#数据读写处理的基类
class getdataBase():
    def __init__(self,gConfig):
        self.gConfig = gConfig
        self.rawshape = self.get_rawshape(self.gConfig)
        self.resizedshape = self.rawshape
        self.cpu_num = self.gConfig['cpu_num']
        self.train_iter=None
        self.test_iter=None
        self.unitestIsOn = self.gConfig['unittestIsOn'.lower()]
        self.ctx = self.gConfig['ctx']

    def load_data(self,*args):
        pass

    def get_rawshape(self,gConfig):
        is_find = False
        #dataset_name = re.findall('get(.*)Data', self.__class__.__name__).pop().lower()
        #assert dataset_name in gConfig['datasetlist'], \
        #    'datasetlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
        #    (gConfig['datasetlist'],dataset_name,self.__class__.__name__)
        dataset_name = self.get_dataset_name(gConfig)
        for key in gConfig:
            if key.find('.') >= 0:
                dataset_key = re.findall('(.*)\.',key).pop().lower()
                if dataset_key == dataset_name:
                    is_find = True
        if is_find == False:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        return [gConfig[dataset_name+'.channels'],gConfig[dataset_name+'.dim_x'],gConfig[dataset_name+'.dim_y']]

    def get_dataset_name(self,gConfig):
        dataset_name = re.findall('get(.*)Data', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['datasetlist'], \
            'datasetlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'], dataset_name, self.__class__.__name__)
        return dataset_name

    # 装饰器，用于在unitest模式下，只返回一个数据，快速迭代
    @staticmethod
    def getdataForUnitest(getdata):
        def wapper(self, batch_size):
            if self.unitestIsOn == True:
                # 仅用于unitest测试程序
                def reader():
                    for (X, y) in getdata(self,batch_size):
                        yield (X, y)
                        break
                return reader()
            else:
                return getdata(self,batch_size)
        return wapper

    @getdataForUnitest.__get__(object)
    def getTrainData(self,batch_size):
        return self.train_iter

    @getdataForUnitest.__get__(object)
    def getTestData(self,batch_size):
        return self.test_iter

    @getdataForUnitest.__get__(object)
    def getValidData(self,batch_size):  # ,batch_size,num_steps):
        pass

    def endProcess(self):
        pass

class getdataBaseM(getdataBase):
    def __init__(self,gConfig):
        super(getdataBaseM,self).__init__(gConfig)
        self.dataset_name = self.get_dataset_name(self.gConfig)
        self.data_path = os.path.join(self.gConfig['data_directory'],self.dataset_name)
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        from mxnet.gluon import data as gdata
        self.dataset_selector = {
            'mnist': gdata.vision.MNIST,
            'fashionmnist': gdata.vision.FashionMNIST,
            'cifar10': gdata.vision.CIFAR10
        }
        self.load_data(resize=self.resize,root=self.data_path)

    def load_data(self,resize=None,root=""):
        from mxnet.gluon import data as gdata
        root = os.path.expanduser(root)
        transformer = []
        if resize is not None and resize != 0:
            transformer += [gdata.vision.transforms.Resize(resize)]
            self.resizedshape = [self.rawshape[0],resize,resize]
        transformer += [gdata.vision.transforms.ToTensor()]
        transformer = gdata.vision.transforms.Compose(transformer)
        #train_data = gdata.vision.MNIST(root=root,train=True)
        #test_data = gdata.vision.MNIST(root=root, train=False)
        train_data = self.dataset_selector[self.dataset_name](root=root,train=True)
        test_data = self.dataset_selector[self.dataset_name](root=root,train=False)
        num_workers = 0 if sys.platform.startswith('win32') else self.cpu_num
        self.train_iter = gdata.DataLoader(train_data.transform_first(transformer),
                                           self.batch_size, shuffle=True,
                                           num_workers=num_workers)
        self.test_iter = gdata.DataLoader(test_data.transform_first(transformer),
                                          self.batch_size, shuffle=False,
                                          num_workers=num_workers)

class getdataBaseH(getdataBase):
    def __init__(self,gConfig):
        super(getdataBaseH,self).__init__(gConfig)
        self.dataset_name = self.get_dataset_name(self.gConfig)
        self.data_path = os.path.join(self.gConfig['data_directory'], self.dataset_name)
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        from torchvision import datasets
        self.dataset_selector={
            'mnist':datasets.MNIST,
            'fashionmnist':datasets.FashionMNIST,
            'cifar10':datasets.CIFAR10
        }
        self.load_data(resize=self.resize, root=self.data_path)

    def load_data(self,resize=None,root=""):
        from torch import utils
        from torchvision import transforms
        root = os.path.expanduser(root)
        transformer = []
        if resize is not None and resize != 0:
            transformer += [transforms.Resize(resize)]
            self.resizedshape = [self.rawshape[0],resize,resize]
        transformer += [transforms.ToTensor()]
        transformer += [transforms.Normalize((0.1307,),(0.3081,))]
        transformer = transforms.Compose(transformer)
        #train_data = datasets.MNIST(root=root,train=True,download=True,transform=transformer)
        #test_data = datasets.MNIST(root=root, train=False,download=True,transform=transformer)
        train_data = self.dataset_selector[self.dataset_name](root=root,train=True,download=True,transform=transformer)
        test_data = self.dataset_selector[self.dataset_name](root=root,train=True,download=True,transform=transformer)
        num_workers = 0 if sys.platform.startswith('win32') else self.cpu_num
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.ctx == 'gpu' else {'num_workers':num_workers}
        self.train_iter = utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True,**kwargs)
        self.test_iter = utils.data.DataLoader(test_data,batch_size=self.batch_size,shuffle=True,**kwargs)
