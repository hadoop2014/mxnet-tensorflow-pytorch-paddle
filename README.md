# mxnet-tensorflow-pytorch-paddle
多个深度学习框架的组合，有助于理解在不同深度学习框架写模型的差异．

本项目环境可通过如下命令创建：
conda env create -f environment.yaml
请事先安装anaconda ,推荐版本4.7.0
另外，对于paddlepaddle的GPU版本，需要额外安装独立的cuda版本

另外，用户需手工创建如下目录：
１）logging_directory   用于存放log文件，包括mxnet,tensorflow,paddle,pytorch的可视化组件的输出，对于tensorflow和pytorch，可运行tensorboard --logdir  logging_directory/tensorflow   或者　tensorboard --logdir  logging_directory/pytorch，　对于paddle　，可运行visualdl --logdir  logging_directory/paddle
２）working_directory 用于存放模型持久化文件，所有变量和模型的持久化文件，如tensorflow的ckeckpoint文件存放于此目录．

