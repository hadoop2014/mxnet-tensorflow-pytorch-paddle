import unittest
import execute
from datafetch import getConfig
import subprocess
import os
import sys
import time

class ExecuteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_allmodel(self):
        gConfig = getConfig.get_config()
        logging_directory = gConfig['logging_directory']
        unittest_logfile = gConfig['unittest_logfile']
        unittest_logfilename = os.path.join(logging_directory,unittest_logfile)
        test_starttime = time.time()
        result = []
        unittestIsOn = True #unittest模式下设置unittestIsOn = True
        assert unittestIsOn == True,\
            'Now in unittest mode,the unittestIsOn must be True which is in configbase.txt'

        for taskName in gConfig['tasknamelist']:
            for framework in gConfig['frameworklist']:
                for dataset in gConfig['datasetlist']:
                    for mode in gConfig['modelist']:
                        if execute.validate_parameter(taskName,framework,dataset,mode,gConfig) == False:
                            continue
                        print('taskName=%s,framework=%s,dataset=%s'%(taskName,framework,dataset))
                        #execute.trainStart(gConfig,taskName,framework,dataset)
                        command = 'python ' 'execute.py' + \
                                  ' ' + str(taskName) + ' ' + str(framework) + ' ' + str(dataset) + ' ' \
                                  + str(mode) + ' ' + str(unittestIsOn)
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
