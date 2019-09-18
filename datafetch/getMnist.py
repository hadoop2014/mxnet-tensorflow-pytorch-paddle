from datafetch.getBaseClass import  *
import sys

class getMnistDataH(getdataBase):
    def __init__(self,gConfig):
        super(getMnistDataH,self).__init__(gConfig)
        self.data_path = os.path.join(self.gConfig['data_directory'], 'mnist')
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
        train_data = datasets.MNIST(root=root,train=True,download=True,transform=transformer)
        test_data = datasets.MNIST(root=root, train=False,download=True,transform=transformer)
        num_workers = 0 if sys.platform.startswith('win32') else self.cpu_num
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.ctx == 'gpu' else {'num_workers':num_workers}
        self.train_iter = utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True,**kwargs)
        self.test_iter = utils.data.DataLoader(test_data,batch_size=self.batch_size,shuffle=True,**kwargs)

class getMnistDataM(getdataBase):
    def __init__(self,gConfig):
        super(getMnistDataM, self).__init__(gConfig)
        self.data_path = os.path.join(self.gConfig['data_directory'],'mnist')
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        self.load_data(resize=self.resize,root=self.data_path)

    def load_data(self,resize=None,root=os.path.join('~/.mxnet','datasets','mnist')):
        from mxnet.gluon import data as gdata
        root = os.path.expanduser(root)
        transformer = []
        if resize is not None and resize != 0:
            transformer += [gdata.vision.transforms.Resize(resize)]
            self.resizedshape = [self.rawshape[0],resize,resize]
        #transformer += [gdata.vision.transforms.ToTensor()]
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
class_selector = {
    "mxnet":getMnistDataM,
    "tensorflow":getMnistDataM,
    "pytorch":getMnistDataH,
    "paddle":getMnistDataM
}

def create_model(gConfig):
    #getdataClass=getMnistData(gConfig=gConfig)
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
