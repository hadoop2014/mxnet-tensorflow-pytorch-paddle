from  datafetch.getBaseClass import *
import numpy as np
from mxnet import  nd
import mxnet as mx
import zipfile
import random


class getLyricDataM(getdataBase):
    def __init__(self,gConfig):
        super(getLyricDataM,self).__init__(gConfig)
        self.data_path = self.gConfig['data_directory']
        self.filename = self.gConfig['lybric_filename']
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        self.num_steps = self.gConfig['num_steps']
        self.ctx = self.get_ctx(gConfig['ctx'])
        self.load_data(self.filename,root=self.data_path)

    def get_ctx(self,ctx):
        assert ctx in self.gConfig['ctxlist'], 'ctx(%s) is invalid,it must one of %s' % \
                                                               (ctx, self.gConfig['ctxlist'])
        if ctx == 'gpu':
            ctx = mx.gpu(0)
        else:
            ctx = mx.cpu(0)
        return ctx

    def load_data(self,filename,root):
        root = os.path.expanduser(root)
        with zipfile.ZipFile(os.path.join(root,filename)) as zin:
            filename = re.findall('(.*).zip',filename).pop()
            with zin.open(filename) as f:
                corpus_chars = f.read().decode('utf-8')
        corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
        self.idx_to_char = list(set(corpus_chars))
        self.char_to_idx = dict([(char, i) for i, char in enumerate(self.idx_to_char)])
        self.vocab_size = len(self.char_to_idx)
        self.corpus_indices = [self.char_to_idx[char] for char in corpus_chars]

        self.transformers = [self.fn_onehot]
        self.resizedshape = [self.vocab_size]
        train_nums = int(self.test_percent * self.vocab_size)
        self.train_data = self.corpus_indices[:train_nums] #paddle.dataset.cifar.train10()#gdata.vision.FashionMNIST(root=root,train=True)
        self.test_data = self.corpus_indices[train_nums:]#paddle.dataset.cifar.test10()#gdata.vision.FashionMNIST(root=root,train=False)

    def fn_onehot(self,x):
        return nd.one_hot(x,self.vocab_size)
        #return self.embedding(x)

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
            for (X,y) in reader:
                for transformer in transformers:
                    X = transformer(X)
                yield (X,y)

        return transform_reader

    def data_iter_random(self,corpus_indices, batch_size, num_steps, ctx=None):
        # 减1是因为输出的索引是相应输入的索引加1
        num_examples = (len(corpus_indices) - 1) // num_steps
        epoch_size = num_examples // batch_size
        example_indices = list(range(num_examples))
        random.shuffle(example_indices)

        # 返回从pos开始的长为num_steps的序列
        def _data(pos):
            return corpus_indices[pos: pos + num_steps]

        for i in range(epoch_size):
            # 每次读取batch_size个随机样本
            i = i * batch_size
            batch_indices = example_indices[i: i + batch_size]
            X = [_data(j * num_steps) for j in batch_indices]
            Y = [_data(j * num_steps + 1) for j in batch_indices]
            yield nd.array(X, ctx), nd.array(Y, ctx)

    def data_iter_consecutive(self,corpus_indices, batch_size, num_steps, ctx=None):
        corpus_indices = nd.array(corpus_indices, ctx=ctx)
        data_len = len(corpus_indices)
        batch_len = data_len // batch_size
        indices = corpus_indices[0: batch_size * batch_len].reshape((
            batch_size, batch_len))
        epoch_size = (batch_len - 1) // num_steps
        for i in range(epoch_size):
            i = i * num_steps
            X = indices[:, i: i + num_steps]
            Y = indices[:, i + 1: i + num_steps + 1]
            yield X, Y
    '''
    def getdataForUnitest(self,data_iter):
        if self.unitestIsOn == True:
            #仅用于unitest测试程序
            def reader():
                for X,y in data_iter:
                    yield X,y
                    break
            return reader()
        else:
            return data_iter
    '''
    @getdataBase.getdataForUnitest
    def getTrainData(self,batch_size):
        train_iter = self.data_iter_random(self.train_data,self.batch_size,self.num_steps,self.ctx)
        self.train_iter = self.transform(train_iter,self.transformers)
        return self.train_iter()

    @getdataBase.getdataForUnitest
    def getTestData(self,batch_size):
        test_iter = self.data_iter_consecutive(self.test_data,self.batch_size,self.num_steps,self.ctx)
        self.test_iter = self.transform(test_iter,self.transformers)
        return self.test_iter()

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

class_selector = {
    "mxnet":getLyricDataM,
    "tensorflow":getLyricDataM,
    "pytorch":getLyricDataM,
    "paddle":getLyricDataM
}

def create_model(gConfig):
    #getdataClass=getMnistData(gConfig=gConfig)
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
