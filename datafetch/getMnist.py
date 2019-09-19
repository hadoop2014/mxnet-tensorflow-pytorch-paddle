from datafetch.getBaseClass import  *
import sys

class getMnistDataH(getdataBaseH):
    def __init__(self,gConfig):
        super(getMnistDataH,self).__init__(gConfig)

class getMnistDataM(getdataBaseM):
    def __init__(self,gConfig):
        super(getMnistDataM, self).__init__(gConfig)

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
