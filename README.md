# Machine learning regression algorithm

`针对诸如股票预测，房价预测，客运量，销量预测进行建模和分析，代码仅供学习，不代表任何观点`

## Install

`pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple`

or

`sh install.sh`

## Data

`数据一般有很多地方可以获取，对于股票数据，建议采用优矿官网进行获取`

- data文件夹下是提供的一个股票和一个货运量的数据

## Core

> - core里面提供了一些经典的用于回归的算法，支持如下算法：
>
>   \* [Y ] svr [支持向量回归]
>
>   \* [Y] KNN
>
>   \* [Y] gs算法，即基于网格搜索的算法
>
>   \* [Y] gs_svr
>
>   \* [Y] ga [遗传算法]
>
>   \* [Y] ga_svr
>
>   \* [Y] AdaBoost、、、and so on、、、

## Utils

- Utils 里面提供了一些用于加载数据、画图、预处理等算法的封装

## Examples

- 这个里面提供了两个关于股票和房价预测的可执行例子。来自于互联网。如有侵权，请联系立马撤去。

## Script

> - 执行KNN算法
>
> `python knn_train.py`
>
> 参数：
>
> 1.path是你数据的路径
>
> 2.save是可选的用于保存渲染的图片
>
> <!--需要注意的是utils里面的数据加载函数当中，需要根据需要进行修改相应的代码-->
>
> - 执行svr算法等其他算法通过执行script里面对应脚本就行。

## Note

`从实验过程来看，用于参数优化的算法，如ga和gs搜索过程不尽相同，ga更加的优秀，但是gs收敛的更快，loss平滑。svr算法需要做数据预处理，不然会出现score很低的情况，但是一些bagging的算法却需要标准化等处理。从实验来看，bagging算法要更加的优秀。knn算法依赖数据量以及模型参数，参数当中lr对于收敛非常的重要，大家可以在使用当中大胆的调试相应的参数`

### Performance

### Citation

\- hdkustl@163.com