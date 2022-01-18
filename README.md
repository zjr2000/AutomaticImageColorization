# AutomaticImageColorization
 Automatic Image Colorization。SUSTech CS308 final project

 ## TODO
 1. Evaluate脚本，需要调研相关的evaluate方法，原文中是使用分类的Acc来评价的，这里最好还是调研一下colorization
 的评价方法。目前看到的有AuC，可能还需要一些别的。

 2. checkpoint保存，目前考虑两种可能的操作，第一种是加入验证集，保存验证集上loss最低的checkpoint。第二种是隔一定
 数量的step跑一下evaluate，在测试集或验证集跑，保存evaluate结果最好的checkpoint。目前看到的项目中都有使用的，不
 确定那种是好的实践，后续需要交流下。

 3. 可视化，evaluate之后最好能将测试集跑后的结果save下来，方便后续展示。因为CIFAR10这个数据集已经序列化了，所以GT
 也要手动处理下。后续考虑补充一个接口。

 4. 调试模型

 5. 补充实验，一个初步的想法是，将global分类网络替换为现有的在大规模数据集上预训练好的网络，fine-tune之后看下结果，暂时
 没有想到别的实验

 6. colorization loss中ruduction使用sum会特别大，跟cls的loss完全不在一个量级，不知道会不会有影响，这个看后面怎么解决一下。
 
 ## Get Started
 ```
 pytorch==1.7.1
 pip install einops
 pip install scikit-image
 ```
## Requirement data
 ```
Download the data from: http://places2.csail.mit.edu/download.html
 ```
 
## Train
 ```
bash train.sh
 ```
 
## Test
 ```
bash eval.sh
 ```
## Colorization
 ```
bash colorization.sh
 ```
