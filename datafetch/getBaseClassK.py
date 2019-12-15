from datafetch.getBaseClass import  *
import os
import sys
import tensorflow as tf

class getdataBaseK(getdataBase):
    def __init__(self,gConfig):
        super(getdataBaseK,self).__init__(gConfig)
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        self.tf = __import__('tensorflow')
        import tensorflow.keras.datasets as datasets
        self.dataset_selector = {
            'mnist': datasets.mnist,
            'fashionmnist': datasets.fashion_mnist,
            'cifar10': datasets.cifar10
        }
        self.load_data(root=os.path.join(os.getcwd(),self.data_path,'.'.join([self.dataset_name,'npz'])))

    def get_transformer(self,train=True):
        #import tensorflow.data as gdata
        #默认情况下train = True和train=False使用的变换相同
        transformer = []
        if self.resize is not None and self.resize != 0:
            transformer += [self.transform_resize(self.resize)]
            self.resizedshape = [self.rawshape[0], self.resize, self.resize]
        transformer += [self.transform_normalize()]
        return transformer

    def load_data(self,root=""):
        #from tensorflow import data as gdata
        root = os.path.expanduser(root)
        train_data,test_data = self.dataset_selector[self.dataset_name].load_data(root)
        #test_data = self.dataset_selector[self.dataset_name].load_data(root)
        train_iter =  tf.data.Dataset.from_tensor_slices(train_data)
        test_iter =  tf.data.Dataset.from_tensor_slices(test_data)
        transformers = self.get_transformer(train=True)
        for transformer in transformers:
            train_iter = train_iter.map(transformer)
        self.train_iter =  train_iter.shuffle(train_data[0].shape[0]).batch(self.batch_size)
        transformers = self.get_transformer(train=False)
        for transformer in transformers:
            test_iter = test_iter.map(transformer)
        self.test_iter = test_iter.batch(self.batch_size)

    def transform_resize(self,resize):
        def resized(x,y):
            if len(x.shape) < 3:
                x = tf.expand_dims(x,-1)
            x = tf.image.resize(x,(resize,resize))
            #x = tf.transpose(x,perm=[2,0,1])
            return (x,y)
        return resized

    def transform_normalize(self):
        def normalized(x,y):
            x = tf.cast(x, tf.float32) / 255.0
            y = tf.cast(y, tf.int64)
            return (x,y)
        return normalized
