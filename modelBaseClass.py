import os
import time
import re
import logging

#深度学习模型的基类
class modelBase():
    def __init__(self,gConfig):
        self.gConfig = gConfig
        self.start_time = time.time()
        self.working_directory = self.gConfig['working_directory']
        self.logging_directory = self.gConfig['logging_directory']
        self.model_savefile = os.path.join(self.working_directory,
                                           self.get_model_name(self.gConfig) + 'model.' + self.gConfig['framework'])
        self.symbol_savefile = os.path.join(self.working_directory,
                                            self.get_model_name(self.gConfig) + 'symbol.' + self.gConfig['framework'])
        self.logging_directory = os.path.join(self.logging_directory, self.gConfig['framework'])
        self.checkpoint_filename = self.get_model_name(self.gConfig)+'.ckpt'
        self.epoch_per_print = self.gConfig['epoch_per_print']
        self.debug_per_steps = self.gConfig['debug_per_steps']
        self.epochs_per_checkpoint = self.gConfig['epochs_per_checkpoint']
        self.batch_size = self.gConfig['batch_size']
        self.debugIsOn = self.gConfig['debugIsOn'.lower()]

        self.losses_train = []
        self.acces_train = []
        self.losses_valid = []
        self.acces_valid = []
        self.losses_test = []
        self.acces_test = []

    def get_model_name(self,gConfig):
        model_name = re.findall('(.*)Model', self.__class__.__name__).pop().lower()
        assert model_name in gConfig['tasknamelist'], \
            'tasknamelist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['tasknamelist'], model_name, self.__class__.__name__)
        return model_name

    def get_net(self):
        pass

    def get_context(self):
        pass

    def get_learningrate(self):
        pass

    def get_globalstep(self):
        pass

    def saveCheckpoint(self):
        pass

    def run_step(self,epoch,train_iter,valid_iter,test_iter,epoch_per_print):
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=None,None,None,None,None,None
        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test

    def run_epoch(self,getdataClass, epoch, output_log=True):
        train_iter = getdataClass.getTrainData(self.batch_size)
        test_iter = getdataClass.getTestData(self.batch_size)
        valid_iter = getdataClass.getValidData(self.batch_size)

        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test = \
            self.run_step(epoch,train_iter,valid_iter,test_iter,self.epoch_per_print)

        if epoch % self.epoch_per_print == 0:
            self.losses_train.append(loss_train)
            self.acces_train.append(acc_train)

            self.losses_valid.append(loss_valid)
            self.acces_valid.append(acc_valid)

            self.losses_test.append(loss_test)
            self.acces_test.append((acc_test))

            check_time = time.time()
            if loss_valid is None:
                print("epoch %d:" % (epoch), "train_time(%depochs)" % self.gConfig['epoch_per_print'],
                      "=%.2f" % (check_time - self.start_time),
                      "\t acc_train = %.4f" % acc_train, "\t loss_train = %.4f" % loss_train,
                      "\t acc_test = %.4f" % acc_test, "\t loss_test = %.4f" % loss_test,
                      #"\t acc_valid = %.4f" % acc_valid, "\t loss_valid = %.4f" % loss_valid,
                      "\t learning_rate = %.6f" % self.get_learningrate(),
                      '\t global_step = %d' % self.get_globalstep(),
                      "  context:%s" % self.get_context())
            elif loss_test is None:
                print("epoch %d:" % (epoch), "train_time(%depochs)" % self.gConfig['epoch_per_print'],
                      "=%.2f" % (check_time - self.start_time),
                      "\t acc_train = %.4f" % acc_train, "\t loss_train = %.4f" % loss_train,
                      #"\t acc_test = %.4f" % acc_test, "\t loss_test = %.4f" % loss_test,
                      "\t acc_valid = %.4f" % acc_valid, "\t loss_valid = %.4f" % loss_valid,
                      "\t learning_rate = %.6f" % self.get_learningrate(),
                      '\t global_step = %d' % self.get_globalstep(),
                      "  context:%s" % self.get_context())
            else:
                print("epoch %d:" % (epoch), "train_time(%depochs)" % self.gConfig['epoch_per_print'],
                      "=%.2f" % (check_time - self.start_time),
                      "\t acc_train = %.4f" % acc_train, "\t loss_train = %.4f" % loss_train,
                      "\t acc_test = %.4f" % acc_test, "\t loss_test = %.4f" % loss_test,
                      "\t acc_valid = %.4f" % acc_valid, "\t loss_valid = %.4f" % loss_valid,
                      "\t learning_rate = %.6f" % self.get_learningrate(),
                      '\t global_step = %d' % self.get_globalstep(),
                      "  context:%s" % self.get_context())
            self.start_time = check_time
            self.debug_info()
        if epoch % self.epochs_per_checkpoint == 0:
            self.saveCheckpoint()

        return

    def train(self,model_eval,getdataClass,gConfig,num_epochs):

        return self.losses_train,self.acces_train,self.losses_valid,self.acces_valid,\
               self.losses_test,self.acces_test

    def run_trainloss(self,getdataBase,epoch,output_log=False):
        return self.acces_train,self.losses_train

    def run_validloss(self,getdataBase,epoch,output_log=False):
        return self.acces_valid, self.losses_valid

    def run_testloss(self,getdataBase,epoch,output_log=False):
        return self.acces_test, self.losses_test

    def debug_info(self,*kargs):
        pass

    def clear_logging_directory(self,logging_directory):
        assert logging_directory == self.logging_directory ,'It is only clear logging directory, but %s is not'%logging_directory
        files = os.listdir(logging_directory)
        for file in files:
            full_file = os.path.join(logging_directory,file)
            if os.path.isdir(full_file):
                self.clear_logging_directory(full_file)
            else:
                try:
                    os.remove(full_file)
                except:
                   print('%s is not be removed'%full_file)
