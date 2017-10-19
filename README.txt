文件说明：

(1)
data_fb/
data_tvl1/
两者均为训练ucf101时生成的list，按modality('RGB','Flow'),data_item('train','val')和split(1,2,3)切分。

(2)
log/
为每次运行保存的日志，由log.py生成，每次执行程序自动保存。

(3)
ops/
包含一些基本功能性的操作，详见ops/utils.py, 和新建的pytorch layer的定义，比如ops/basic_ops.py

(4)
pyActionRecog/
一些pycaffe的工具

(5)
tf_model_zoo/
包含除了resnet,googlenet等常用网络之外的一些深层网络的定义，down weights等操作

(6)
tmp_result/
将测试结果暂时保存的目录，用于计算mAP等

(7)
config.py
读取utils.config里的配置信息，可将算法涉及到的参数放进utils.config，配合shell脚本直接执行算法

(8)
dataset.py
main.py
test_models.py
models.py
opts.py
基于TSN的训练与测试，在xiong的官方code上有一些改进

(9)
perframe_train.py
perframe_test.py
perframe_dataset.py
models.py
本次实验的baseline部分。

(10)
multiframe_dataset.py
multiframe_train.py
multiframe_test.py
基于TSN的segment-level的训练和测试。正在coding中。

(11)
transforms.py
一些data preprocess 和postprocess的操作

(12)
visualize_tool.py
打印log/里保存的训练、测试日志的训练曲线的方法

(13)
eval_mAP.py
计算tmp_results/下的测试结果的mAP的方法

(14)
tmporal_pool.py
基于LSTM的action recognition的训练代码

