import unittest
import execute
from datafetch import getConfig
import subprocess
import os
import sys
import time

def validate_parameter(taskName,framework,dataset):
    if taskName == 'regression':
        if framework != 'mxnet':
            return False
        else:
            if dataset != 'houseprice':
                return False
    else:
        if dataset == 'houseprice':
            return False


    if taskName == 'rnn':
        if framework != 'mxnet':
            return False
        else:
            if dataset != 'lyric':
                return False
    else:
        if dataset == 'lyric':
            return False

    if taskName == 'alexnet':
        pass
    else:
        if framework == 'paddle':
            return False
        else:
            pass

    if taskName == 'capsnet':
        if framework != 'tensorflow':
            return False
        else:
            pass
    else:
        pass

    if taskName == 'lstm':
        return False

    if taskName != 'vgg':
        #用于指定测试某个模型
        pass

    return True

class ExecuteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_allmodel(self):
        gConfig = getConfig.get_config()
        logging_directory = gConfig['logging_directory']
        unittest_logfile = gConfig['unittest_logfile']
        unittest_logfilename = os.path.join(logging_directory,unittest_logfile)
        unitestIsOn = gConfig['unittestIsOn'.lower()]
        test_starttime = time.time()
        result = []
        assert unitestIsOn == True,\
            'Now in unittest mode,the unitestIsOn must be True which is in configbase.txt'
        for taskName in gConfig['tasknamelist']:
            for framework in gConfig['frameworklist']:
                for dataset in gConfig['datasetlist']:
                    if validate_parameter(taskName,framework,dataset) == False:
                        continue
                    print('taskName=%s,framework=%s,dataset=%s'%(taskName,framework,dataset))
                    #execute.trainStart(gConfig,taskName,framework,dataset)
                    command = 'python ' 'execute.py' + \
                              ' ' + str(taskName) + ' ' + str(framework) + ' ' + str(dataset) +' ' + str(unitestIsOn)
                    #with open(unittest_logfilename,'a',encoding='utf-8') as f:
                        #CompletedProcessObject=subprocess.run(command,shell=True,stdout=f,timeout=100,check=False,
                        #                                      universal_newlines=True)
                    CompletedProcessObject = subprocess.run(command, shell=True, stdout=sys.stdout,stderr=sys.stderr)
                    result.append(command+',testtimes = %.2f'%(time.time()-test_starttime))
                    test_starttime =time.time()
                    self.assertTrue(CompletedProcessObject,True)
                    self.assertEqual(CompletedProcessObject.returncode,0)

        for line in result:
            print(line)


if __name__ == '__main__':
    unittest.main()
