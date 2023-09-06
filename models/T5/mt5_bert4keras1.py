import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] ="0"    #这里是gpu的序号，指定使用的gpu对象
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
