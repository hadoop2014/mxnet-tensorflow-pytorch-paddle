from datafetch.getBaseClassH import  *
from datafetch.getBaseClassM import *

class getFashionMnistDataH(getdataBaseH):
    def __init__(self,gConfig):
        super(getFashionMnistDataH,self).__init__(gConfig)


class getFashionMnistDataM(getdataBaseM):
    def __init__(self,gConfig):
        super(getFashionMnistDataM, self).__init__(gConfig)


class_selector = {
    "mxnet":getFashionMnistDataM,
    "tensorflow":getFashionMnistDataM,
    "pytorch":getFashionMnistDataH,
    "paddle":getFashionMnistDataM,
    "keras": getFashionMnistDataM
}


def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
