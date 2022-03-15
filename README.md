# Deep-Learning-Part-of-Gradua~~t~~ion-Internship

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
        NUM_EPOCHS=30
        num_class=7
        save_freq=3

        continue_train=cp_0032.ckpt

        data_root=./Data/Data
        pre_model_dir=./pre_model/
        test_dir=./Data/Data_test
        log_dir=./logs/fit
        reports=./reports
        model_dir=./model
        visual_show_images=visual_show_images
        plot_model_dir=./image_of_model
        save_in_training=./model_ckpt

        model=restnet50
        pre_model=tf2-preview_mobilenet_v2_classification_4
        train=True
   ```
   
2. change the current path to the project

3. 

   需要训练模型

   1. `python main.py --modle=toy_res_net` # 使用自己写的模型

        如果使用的是keras模型， 比如resnet101, 那么他的保存路径是`~/.keras/datasets/example.txt`和`/home/andy/.keras/models/example.h5`，方便网络不好的时候自己存入
        
   2. `python main.py --modle=mobile_v3_transfer_model --pre_model=bit_m-r50x1_imagenet21k_classification_1 `#[迁移学习模型](https://tfhub.dev/s?module-type=image-classification)

   3. `python main.py`  默认使用的是

   4. 后台运行

        1. 启动 `nohup python -u main.py > daemon/name.log 2>&1 & echo $! > daemon/run.pid`
        2. 停止 `[[ -f daemon/run.pid ]] && kill $(cat daemon/run.pid)`

   使用已经存在的模型

   1. `python main.py --model=sequential_1-1646976564 --train=False`

   > modle 在models中定义的函数 或者在models中存在的文件夹
   > pre_model 在pre_model中的文件夹名称

4. tensorboard

   1.  ` tensorboard --logdir logs/fit/`

      tensorboard 并不是很熟练，没能充分利用他的功能，需要完善。 比如目前所有的训练日志都存放到了fit中，没有使用不同的模型
   
5. gpu支持

   1. 学校机房的电脑不是我自己用的，我也不太理解如何配置支持gpu，为了更加保险，选择使用docker

      1. install docker

      2. install nvidia-docker
      
         ```bash
         # Add the package repositories
         $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
         $ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
         $ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
         
         $ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
         $ sudo systemctl restart docker
         ```
      
         Install nvidia-docker2 and reload the Docker daemon configuration
         
         ```bash
         $sudo apt-get install -y nvidia-docker2
         $sudo pkill -SIGHUP dockerd
         ```
         
         测试
         
         ```bash
         docker run --runtime=nvidia -rm nvidia/cuda:9.0-base nvidia-smi
         ```
         
         下载tensorflow
         
         在docker/gpu下运行`docker build -t tf2-gpu:v1`
         
         在docker/gpu下运行`docker-compose up`
         
         运行的模型在docker-compose 中修改需要运行的参数,或者在.env中修改

# 模型英文名字

   `factory   farmland   forest   grassland   parking  'residential area'   water	`
