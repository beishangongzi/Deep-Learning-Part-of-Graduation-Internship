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

   ```bash
   batch_size=32
   img_height=224
   img_width=224
   NUM_EPOCHS=32
   num_class=7
   save_freq=1
   data_root="./Data/Data"
   pre_model="./pre_model/"
   test_dir="./Data/Data_test"
   log_dir="./logs/fit"
   reports="./reports"
   model="./model"
   visual_show_images="visual_show_images"
   plot_model=./image_of_model
   
   
   model=mobile_v3_transfer_model
   pre_model=bit_m-r50x1_imagenet21k_classification_1
   train=True
   ```

2. change the current path to the project

3. 

   需要训练模型

   1. `python main.py --modle=toy_res_net` # 使用自己写的模型
   2. `python main.py --modle=mobile_v3_transfer_model --pre_model=bit_m-r50x1_imagenet21k_classification_1 `#[迁移学习模型](https://tfhub.dev/s?module-type=image-classification)
   3. `python main.py`  默认使用的是

   使用已经存在的模型

   1. `python main.py --model=sequential_1-1646976564 --train=False`

   > modle 在models中定义的函数 或者在models中存在的文件夹
   > pre_model 在pre_model中的文件夹名称

4. tensorboard

   1.  ` tensorboard --logdir logs/fit/`

      tensorboard 并不是很熟练，没能充分利用他的功能，需要完善。 比如目前所有的训练日志都存放到了fit中，没有使用不同的模型



# 模型英文名字

   `factory   farmland   forest   grassland   parking  'residential area'   water	`
