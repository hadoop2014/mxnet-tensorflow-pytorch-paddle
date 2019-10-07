from datafetch.getBaseClassM import  *


class getHotdogDataM(getdataBaseM):
    def __init__(self,gConfig):
        super(getHotdogDataM, self).__init__(gConfig)

    def get_transformer(self,train=True):
        import mxnet.gluon.data as gdata
        transformer = []
        normalize = gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train == True:
            if self.resize == 0 :
                self.resize = 244 #如果没有设置resize大小，默认使用224的resize
            if self.resize is not None and self.resize != 0:
                transformer += [gdata.vision.transforms.RandomResizedCrop(self.resize)]
                self.resizedshape = [self.rawshape[0], self.resize, self.resize]
            transformer += [gdata.vision.transforms.RandomFlipLeftRight()]
            transformer += [gdata.vision.transforms.ToTensor()]
            transformer += [normalize]
            transformer = gdata.vision.transforms.Compose(transformer)
        else:
            if self.resize == 0:
                self.resize = 244 #如果没有设置resize大小，默认使用224的resize
            if self.resize is not None and self.resize != 0:
                assert self.resize <= 256 and self.resize > 0,'self.resize %d is invalid,it must be (0 < resize <= 256)!'
                transformer += [gdata.vision.transforms.Resize(256)]  #默认使用
                transformer += [gdata.vision.transforms.CenterCrop(self.resize)]
                self.resizedshape = [self.rawshape[0], self.resize, self.resize]
            transformer += [gdata.vision.transforms.ToTensor()]
            transformer += [normalize]
            transformer = gdata.vision.transforms.Compose(transformer)
        return transformer

class_selector = {
    "mxnet":getHotdogDataM,
    "tensorflow":getHotdogDataM,
    "pytorch":getHotdogDataM,
    "paddle":getHotdogDataM
}

def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
