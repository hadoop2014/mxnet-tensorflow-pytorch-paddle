from datafetch.getBaseClass import  *
from mxnet.gluon import data as gdata
#import tensorflow as tf
import sys

class getMnistData(getdataBase):
    def __init__(self,gConfig):
        super(getMnistData, self).__init__(gConfig)
        self.data_path = os.path.join(self.gConfig['data_directory'],'mnist')
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        self.load_data(resize=self.resize,root=self.data_path)

    def load_data(self,resize=None,root=os.path.join('~/.mxnet','datasets','mnist')):
        root = os.path.expanduser(root)
        transformer = []
        if resize is not None and resize != 0:
            transformer += [gdata.vision.transforms.Resize(resize)]
            self.resizedshape = [self.rawshape[0],resize,resize]
        transformer += [gdata.vision.transforms.ToTensor()]
        transformer = gdata.vision.transforms.Compose(transformer)
        train_data = gdata.vision.MNIST(root=root,train=True)
        test_data = gdata.vision.MNIST(root=root, train=False)
        num_workers = 0 if sys.platform.startswith('win32') else self.cpu_num
        self.train_iter = gdata.DataLoader(train_data.transform_first(transformer),
                                           self.batch_size, shuffle=True,
                                           num_workers=num_workers)
        self.test_iter = gdata.DataLoader(test_data.transform_first(transformer),
                                          self.batch_size, shuffle=False,
                                          num_workers=num_workers)


def create_model(gConfig):
    getdataClass=getMnistData(gConfig=gConfig)
    return getdataClass
