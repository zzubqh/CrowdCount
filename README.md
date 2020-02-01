# CrowCount
基于mscnn的人群数量估计

## 项目简介
1. 一个人群密度估计的项目，检测的图片包括不包含任何人的图片、只有有限数量人的图片和密度很高的图片；
2. 使用keras框架复现mscnn论文；
3. 项目中包含一个使用c#开发的人群密度等级标记工具，一个使用vgg16作为特征提取网络的密度分类模型，一个用于人数估计的mscnn模型

## 密度图生成
对标注过行人位置的图片使用高斯滤波处理后得到类似热力图的形式。滤波处理是因为在计算损失函数的时候，由于每个人只占一个像素点的密度图特别稀疏，这样会导致模型收敛到全0状态。高斯滤波后，一方面呈现出来比较直观，另外在一定程度上解决了密度图的稀疏问题。由于人群密度高的时候，人群分布在图片上存在遮挡变形等，故在高斯滤波的时候采用自适应核的方式，高斯核的方差与人头间的距离成比例。详见data.py中的get_densemap()函数。

## 数据集说明
提供三个数据集下载地址，ShanghaiTech和Mall是做人群估计的常用数据集，
- ShanghaiTech Dataset
  - [下载地址](https://pan.baidu.com/s/1hseCEr7v7828DFLj8BQWAw)
- Mall Dataset
  - [下载地址](https://pan.baidu.com/s/1gjFTqcO1gvQnYIDs8CDIJw)
- 本次所使用数据集
  - [下载地址](https://pan.baidu.com/s/1T5EfBovMnpe4meIYcXSa8w)
- 数据集说明
  - 提供的是我的百度网盘地址，如果下载有问题可以直接联系我。邮箱：zzubqh@gmail.com
  - 本次使用数据集压缩包里有两个压缩文件，Denselevel.rar为标注好的图片密度等级，分成3个等级 0：不包含任何人的图片；1：包含的人员数小于100；2：包含的人员数大于100。img文件夹中是图片，dense_gt.mat是标签文件
  - Crowdata.rar为标注好的人头位置信息的数据集。img文件夹中是图片，crow.mat是标签文件
  - 项目中包含了一个使用vs2007开发的CrawDenseTool项目，用来给图片的密度等级打标签。修改源码中的图片目录地址然后重新编译后即可使用，快捷键A,D为上一张/下一张，方向键"上/下"为标签选择，"保存"按钮保存的文件以当前时间命名。
  - 项目中的tools.py文件，函数create_denselevelLabel()提供了将CrawDenseTool项目保存的文件转换成标签文件的功能；函数create_crowLabel()提供了将labelme工具生成的json文件转换成标签文件的功能。
  - 生成自己的数据集时，首先使用labelme工具对图片中的人头进行标记，然后利用函数create_crowLabel()函数转换即可。  

## 环境配置
- 基于Python3.6
- 需要第三方包已在[requirements](/requirements.txt)列出
	- 切换到requirements文件所在目录，执行命令`pip install -r requirements.txt`即可配置环境
- 脚本运行说明
	- 训练
		- 命令行执行
			- `python train.py -b 8`		
	- 测试
    		- 命令行执行
    			- `python test.py`		
## 模型构建
- 密度等级分类模型，详见vggmodel.py
    - 使用vgg16作为特征提取网络，后接3个全连接层用于分类。密度等级0,1,2使用one-hot编码形式
- 人群密度估计使用mscnn模型，详见model.py 
    - 注意
        - **输出层千万不能使用Relu，否则输出会导致预测均为0,并且loss确实在不断降低！**	
## 模型训练
- 训练好的模型地址
  - [下载地址](https://pan.baidu.com/s/105FM8Di3MqsWsN6-l-S2vQ)，提取码：i2dn
  - 将下载后的模型存入models文件夹下
- 训练数据集
    - 本次主要在自己的数据集上训练，若要在其他数据集上，参考data.py即可。
- 训练效果展示（模型简单训练5轮）
    - crowdataset
        - 在crowdataset上随机5张图片进行密度图预测，结果如下。
        - ![](/result/res.png)
    
## 补充说明
- 关于代码细节和相关理论欢迎在我的[博客](https://blog.csdn.net/qq_36810544)下留言。
- 完整代码已经上传到我的Github，欢迎Star或者Fork。
- 如有错误，欢迎指正。
