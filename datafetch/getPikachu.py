from datafetch.getBaseClassM import  *

class getPikachuDataM(getdataBaseM):
    def __init__(self,gConfig):
        super(getPikachuDataM, self).__init__(gConfig)

    def load_data(self,root=""):
        from mxnet import image
        root = os.path.expanduser(root)
        train_data = self.dataset_selector[self.dataset_name](root=root,train=True)
        test_data = self.dataset_selector[self.dataset_name](root=root,train=False)
        self.resizedshape = self.rawshape
        #transformer = self.get_transformer(train=True)
        self.train_data = image.ImageDetIter(
            path_imgrec=train_data.path_rec,
            path_imgidx=train_data.path_idx,
            batch_size=self.batch_size,
            data_shape=tuple(self.resizedshape),
            shuffle=True,  # 以随机顺序读取数据集
            rand_crop=1,  # 随机裁剪的概率为1
            min_object_covered=0.95,
            max_attempts=200)
        self.test_data = image.ImageDetIter(
            path_imgrec=test_data.path_rec,
            batch_size=self.batch_size,
            data_shape=tuple(self.resizedshape),
            shuffle=False)

    def transform(self,reader):
        def transform_reader():
            for batch in reader:
                X = batch.data[0]
                y = batch.label[0]
                yield (X,y)
        return transform_reader

    @getdataBase.getdataForUnittest
    def getTrainData(self,batch_size):
        self.train_iter = self.transform(self.train_data)
        return self.train_iter()

    @getdataBase.getdataForUnittest
    def getTestData(self,batch_size):
        self.test_iter = self.transform(self.test_data)
        return self.test_iter()

class_selector = {
    "mxnet":getPikachuDataM,
    "tensorflow":getPikachuDataM,
    "pytorch":getPikachuDataM,
    "paddle":getPikachuDataM
}

def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass