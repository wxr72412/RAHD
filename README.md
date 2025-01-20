# Coder for Journal of South China University of Technology (Natural Science Edition)
# 代码：华南理工大学学报(自然科学版)

## 1. Datasets 数据集
(1)CELL:真实的基站数据集因合作方的规定无法公开。
(2)PEMS04：交通流量数据1。
(3)PEMS08：交通流量数据2。

## 2. Data preprocessing 数据预预处理
(1) (filepath: \preprocess.py)  
将数据集划分为训练集和测试集。   
(2) (filepath: \node_process.py)  
将节点特征数据处理成不同方法输入的形式。   

## 3. Hyperparameter setting 超参数设置
config.yml

## 4. Our method 本文方法
RAHD (filepath: \RAHD.py) 

## 5. Comparison methods 对比方法
(1) LSTM (filepath: \LSTM.py)   
(2) TCN (filepath: \TCN.py)  
(3) GC-LSTM (filepath: \GC-LSTM.pyTRIP)   
(4) STGCN (filepath: \STGCN.py)  
(5) ASTGCN (filepath: \ASTGCN.py)  
(6) E-LSTM-D (filepath: \E-LSTM-D.py)  

