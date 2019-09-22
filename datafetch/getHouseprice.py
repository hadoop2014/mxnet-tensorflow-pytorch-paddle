from datafetch.getBaseClass import  *
import pandas as pd
from mxnet import nd
from mxnet.gluon import data as gdata
import numpy as np

class getHousepriceDataM(getdataBase):
    def __init__(self,gConfig):
        super(getHousepriceDataM, self).__init__(gConfig)
        self.data_path = self.gConfig['data_directory']
        self.train_file = self.gConfig['train_file']
        self.test_file = self.gConfig['test_file']
        self.k = self.gConfig['k']
        self.load_data( root=self.data_path,train_file=self.train_file,test_file=self.test_file)

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
        return [gConfig[dataset_name+'.dim']]

    def load_data(self,root,train_file,test_file):
        self.train_data = pd.read_csv(self.data_path+self.train_file)
        self.test_data = pd.read_csv(self.data_path + self.test_file)
        #数据预处理，train_data和test_data的第一列是id,对预测无效，去掉；test_data缺少一列，即salesprice列，也即标签列.
        self.all_features = pd.concat((self.train_data.iloc[:,1:-1],self.test_data.iloc[:,1:]))
        #把所有的数字化列进行归一化．
        numeric_features = self.all_features.dtypes[self.all_features.dtypes != 'object'].index
        self.all_features[numeric_features] = self.all_features[numeric_features].apply(
            lambda x:(x - x.mean())/x.std())
        self.all_features[numeric_features] = self.all_features[numeric_features].fillna(0)
        # dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
        self.all_features = pd.get_dummies(self.all_features,dummy_na=True)

    #Ｋ折交叉验证
    def get_k_fold_data(self,k,features,labels):
        assert k > 1
        fold_size = features.shape[0] // k
        X_train,y_train = None,None
        X_valid,y_valid = None,None
        i = np.random.randint(k)
        for j in range(k):
            idx = slice(j * fold_size,(j+1)*fold_size)
            X_part,y_part = features[idx,:],labels[idx]
            if j == i:
                X_valid,y_valid = X_part,y_part
            elif X_train is None:
                X_train,y_train = X_part,y_part
            else:
                X_train = nd.concat(X_train,X_part,dim=0)
                y_train = nd.concat(y_train,y_part,dim=0)
        return X_train,y_train,X_valid,y_valid

    @getdataBase.getdataForUnittest
    def getTrainData(self,batch_size):
        self.num_train = self.train_data.shape[0]
        train_features = nd.array(self.all_features[:self.num_train].values)
        train_labels = nd.array(self.train_data.SalePrice.values).reshape(-1,1)
        self.train_features,self.train_labels,self.valid_features,self.valid_labels = \
            self.get_k_fold_data(self.k,train_features,train_labels)
        train_iter = gdata.DataLoader(gdata.ArrayDataset(self.train_features, self.train_labels), batch_size, shuffle=True)
        return train_iter

    @getdataBase.getdataForUnittest
    def getTestData(self,batch_size = None):
        self.test_features = nd.array(self.all_features[self.num_train:].values)
        self.test_labels = nd.array([None] * len(self.test_features))
        test_iter = gdata.DataLoader(gdata.ArrayDataset(self.test_features,self.test_labels),
                                     batch_size=self.test_features.shape[0])
        return test_iter

    @getdataBase.getdataForUnittest
    def getValidData(self,batch_size = None):
        valid_iter = gdata.DataLoader(gdata.ArrayDataset(self.valid_features,self.valid_labels),
                                      batch_size=self.valid_features.shape[0])
        return valid_iter

class_selector = {
    "mxnet":getHousepriceDataM,
    "tensorflow":getHousepriceDataM,
    "pytorch":getHousepriceDataM,
    "paddle":getHousepriceDataM
}

def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
