# 主目录文件和文件夹说明

```text
└──animeganv2
    ├──readme.md                               
    ├──train.py                                 # 网络训练
    ├──video2anime.py                           # 视频卡通化
    ├──infer.py                                 # 图像卡通化
    ├──_training_dir                            # 每次训练会生成一个以时间戳命名的文件夹，存放每个epoch的模型和测试结果，@加数字表示效果较好的epoch
    ├──data                                     # vgg模型
    ├──dataset                                  # 数据集
    ├──engine                                   # 各种损失函数
    ├──models                                   # 生成器和判别器网络
    ├──others                                   # 杂乱图片
    ├──utils                                    # 图像处理脚本
    └──video                                    # 存放视频文件

```