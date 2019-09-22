import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from threading import Thread
from datafetch import getConfig
import json
import os

check_book = None

def train(model,model_eval,getdataClass,gConfig,taskName,framework,dataset):
    gConfig = gConfig
    if gConfig['unittestIsOn'.lower()] == True:
        num_epochs = 1
    else:
        num_epochs = gConfig['train_num_epoch']
    start_time = time.time()

    print("\n\ntraining %s starting at plat %s use dataset %s, use optimizer %s,ctx=%s,initializer=%s,check_point=%s,"
          "activation=%s...............\n\n"
          %(taskName,framework,dataset,gConfig['optimizer'],gConfig['ctx'],gConfig['initializer'],
            gConfig['ckpt_used'],gConfig['activation']))
    losses_train,acces_train,losses_valid,acces_valid,losses_test,acces_test=\
        model.train(model_eval,getdataClass,gConfig,num_epochs)
    getdataClass.endProcess()
    plotLossAcc(losses_train,acces_train,losses_valid,acces_valid,losses_test,acces_test,gConfig,taskName)
    print('\n\ntraining %s end, time used %.4f'%(taskName,(time.time()-start_time)))

def plotLossAcc(losses_train,acces_train,losses_valid,acces_valid,losses_test,acces_test,gConfig,taskName):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(np.reshape(losses_train,[-1]),'g',label = 'train loss')
    if losses_test[0] is not None:
        ax1.plot(np.reshape(losses_test,[-1]),'r-',label = 'test loss')
    if losses_valid[0] is not None:
        ax1.plot(np.reshape(losses_valid, [-1]), 'r-', label='valid loss')

    ax1.legend()
    ax1.set_ylabel('loss')
    plt.title(taskName,loc='center')

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(np.reshape(acces_train,[-1]),'g',label = 'train accuracy')
    if acces_test[0] is not None:
        ax2.plot(np.reshape(acces_test,[-1]),'r-', label = 'test accuracy')
    if acces_valid[0] is not None:
        ax2.plot(np.reshape(acces_valid, [-1]), 'r-', label='valid accuracy')
    ax2.set_ylabel('accuracy')
    ax2.legend()
    ax2.set_xlabel(format('epochs (per %d)' % gConfig['epoch_per_print']))

    #thread = Thread(target=closeplt, args=(gConfig['pltsleeptime'],))
    #thread.start()
    if gConfig['unittestIsOn'.lower()] == False:
        #在unittest模式下，不需要进行绘图，否则会阻塞后续程序运行
        plt.show()

def closeplt(time):
    sleep(time)
    plt.close()

def getDataset(taskName,framework,gConfig,dataset):
    module = __import__(check_book['datafetch'][dataset],fromlist=(check_book['datafetch'][dataset].split('.')[-1]))
    getdataClass = getattr(module,'create_model')(gConfig)
    return getdataClass


def modelManager(framework,gConfig,dataset,taskName,ckpt_used=False):
    getdataClass = getDataset(taskName, framework, gConfig, dataset)
    module = __import__(check_book[taskName][framework]["model"],
                        fromlist=(check_book[taskName][framework]["model"].split('.')[-1]))
    model = getattr(module,'create_model')(gConfig=gConfig,ckpt_used=ckpt_used,
                                           getdataClass=getdataClass)
    model_eval = model
    return model,model_eval,getdataClass

def get_gConfig(gConfig,taskName,framework,dataset,unittestIsOn):
    global check_book
    if check_book is not None:
        config_file = os.path.join(gConfig['config_directory'],check_book[taskName]['config_file'])
    else:
        raise ValueError('check_book is None ,it may be some error occured when open the checkbook.json!')
    gConfig = getConfig.get_config(config_file)
    #在unitest模式,这三个数据是从unittest.main中设置，而非从文件中读取．
    gConfig['taskName'] = taskName
    gConfig['framework'] = framework
    gConfig['dataset'] = dataset
    gConfig['unittestIsOn'.lower()] = unittestIsOn
    return gConfig

def trainStart(gConfig,taskName,framework,dataset,unittestIsOn):
    if gConfig['mode'] == 'train':
        gConfig = get_gConfig(gConfig,taskName,framework,dataset,unittestIsOn)
        model, model_eval, getdataClass = modelManager(framework, gConfig, dataset, taskName=taskName,
                                                       ckpt_used=gConfig['ckpt_used'])
        train(model, model_eval, getdataClass, gConfig, taskName, framework,dataset)
    elif gConfig['mode'] == 'server':
        raise ValueError('Sever Usage:python3 app.py')

def set_check_book(gConfig):
    check_file = os.path.join(gConfig['config_directory'], gConfig['check_file'])
    global check_book
    if os.path.exists(check_file):
        with open(check_file, encoding='utf-8') as check_f:
            check_book = json.load(check_f)
    else:
        raise ValueError("%s is not exist,you must create first!" % check_file)


def validate_parameter(taskName,framework,dataset,gConfig):
    assert taskName in gConfig['tasknamelist'], 'taskName(%s) is invalid,it must one of %s' % \
                                                (taskName, gConfig['tasknamelist'])
    assert framework in gConfig['frameworklist'], 'framework(%s) is invalid,it must one of %s' % \
                                                  (framework, gConfig['frameworklist'])
    assert dataset in gConfig['datasetlist'], 'dataset(%s) is invalid,it must one of %s' % \
                                              (dataset, gConfig['datasetlist'])
    global check_book
    set_check_book(gConfig)
    return check_book[taskName][framework][dataset]

def main():
    gConfig = getConfig.get_config()
    if len(sys.argv) > 1 :
        if len(sys.argv) == 5:
            #该模式为从unittest.main调用
            unittestIsOn = bool(sys.argv[4])
            assert unittestIsOn == True , \
                'Now in unittest mode, the num of argvs must be 5 whitch is taskName, framework,dataset and unittestIsOn'
        else:
            #该模式为从python -m 方式调用
            unittestIsOn = gConfig['unittestIsOn'.lower()]
        # print(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],len(sys.argv))
        taskName = sys.argv[1]
        framework = sys.argv[2]
        dataset = sys.argv[3]
    else:
        #该模式为从pycharm调用
        unittestIsOn = gConfig['unittestIsOn'.lower()]
        assert unittestIsOn == False, \
            'Now in training mode,unitestIsOn must be False whitch in configbase.txt'
        taskName = gConfig['taskname']
        framework = gConfig['framework']
        dataset = gConfig['dataset']

    if validate_parameter(taskName,framework,dataset,gConfig) == True:
        trainStart(gConfig, taskName, framework, dataset,unittestIsOn)
    else:
        raise ValueError("(%s %s %s) is not supported now!"%(taskName,framework,dataset))

if __name__=='__main__':
    main()


