# Deep-Learning-Part-of-Graduation-Internship
这是毕业实习深度学习部分

任务要求：
  1. 7种地物分类。类似于猫狗大战

照片是我们自己在天地图截图的。规格：50m分辨率， 200 * 200 * 3像素

现在实现的方法有：
  1. 
  2. 


## 使用方法

    1. premodle 放的是需要使用的训练好的模型
    2. model 放的是训练好的模型
    3. log 存放的是训练日志
    4. reports 存放的是混淆矩阵
    5. visual_show_images 存放的是结果展示

## 启动方法
    1. 在当前路径下建立文件`.env`
        内容
        ```shell

        batch_size=32
        img_height=224
        img_width=224
        NUM_EPOCHS=10
        num_class=7
        data_root="./Data/Data"

        model_dir="./pre_model/"
        test_dir="./Data/Data"
        log_dir="./logs/fit"
        ```
    1. change the current path to the project
    2. 
    `python main.py --modle=toy_res_net`
    `python main.py --modle=mobile_v3_transfer_model --pre_model=bit_m-r50x1_imagenet21k_classification_1`
    `python main.py`

    > modle 在models中定义的函数
    > pre_model 在pre_model中的文件夹名称
