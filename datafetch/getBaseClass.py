import os
import re

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
        dataset_name = re.findall('get(.*)Data', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['datasetlist'], \
            'datasetlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'],dataset_name,self.__class__.__name__)
        for key in gConfig:
            if key.find('.') >= 0:
                dataset_key = re.findall('(.*)\.',key).pop().lower()
                if dataset_key == dataset_name:
                    is_find = True
        if is_find == False:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        return [gConfig[dataset_name+'.channels'],gConfig[dataset_name+'.dim_x'],gConfig[dataset_name+'.dim_y']]

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

