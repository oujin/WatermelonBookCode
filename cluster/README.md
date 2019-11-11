# 聚类算法
1. `kmeans.py`、`gmm.py`、和`dbscan.py`分别实现了三种聚类算法。
2. `data_gen.py`用于生成人工数据集，已生成三个数据集`1.npy`、`2.npy`、`3.npy`保存在`data`目录下。
3. `distance.py`里包含多个距离度量函数，`estimate.py`包含了对算法性能进行量化评估的函数。 
4. `art_data_test.py`是在人工数据集上的实验。
5. `img_seg_test.py`是在分割图像数据上的实验。
6. 程序在Win10上用python3.7编写，需要的依赖包见`requirements.txt`，使用命令
    ```bash
    $ pip install -r requirements.txt
    ```
    安装。
7. `data`文件夹的`*.jpg`是实验使用的图像，类标签存在相应的`*.txt`里。
8. 运行如下两行命令分别进行实验：
    ```bash
    $ python art_data_test.py
    $ python img_seg_test.py
    ```