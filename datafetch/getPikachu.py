from datafetch.getBaseClassM import  *

class getPikachuDataM(getdataBaseM):
    def __init__(self,gConfig):
        super(getPikachuDataM, self).__init__(gConfig)

    def load_data(self,root=""):
        from mxnet.gluon import data as gdata
        from mxnet import image
        root = os.path.expanduser(root)
        train_data = self.dataset_selector[self.dataset_name](root=root,train=True)
        test_data = self.dataset_selector[self.dataset_name](root=root,train=False)
        #num_workers = 0 if sys.platform.startswith('win32') else self.cpu_num
        self.resizedshape = self.rawshape
        #transformer = self.get_transformer(train=True)
        self.train_iter = image.ImageDetIter(
            #path_imgrec=os.path.join(data_dir, 'train.rec'),
            #path_imgidx=os.path.join(data_dir, 'train.idx'),
            path_imgrec=train_data.path_rec,
            path_imgidx=train_data.path_idx,
            batch_size=self.batch_size,
            #data_shape=(3, edge_size, edge_size),  # 输出图像的形状
            data_shape=tuple(self.resizedshape),
            shuffle=True,  # 以随机顺序读取数据集
            rand_crop=1,  # 随机裁剪的概率为1
            min_object_covered=0.95,
            max_attempts=200)
        self.test_iter = image.ImageDetIter(
            #path_imgrec=os.path.join(data_dir, 'val.rec'),
            path_imgrec=test_data.path_rec,
            batch_size=self.batch_size,
            #data_shape=(3, edge_size, edge_size),
            data_shape=tuple(self.resizedshape),
            shuffle=False)

class_selector = {
    "mxnet":getPikachuDataM,
    "tensorflow":getPikachuDataM,
    "pytorch":getPikachuDataM,
    "paddle":getPikachuDataM
}

def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
