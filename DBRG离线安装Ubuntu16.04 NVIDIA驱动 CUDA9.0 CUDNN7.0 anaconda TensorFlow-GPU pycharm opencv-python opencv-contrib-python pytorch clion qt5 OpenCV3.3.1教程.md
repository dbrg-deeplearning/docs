# DBRG离线安装Ubuntu16.04 NVIDIA驱动 CUDA9.0 CUDNN7.0 anaconda TensorFlow-GPU pycharm opencv-python opencv-contrib-python pytorch clion qt5 OpenCV3.3.1教程
## 1、格式化原Ubuntu分区
https://jingyan.baidu.com/article/295430f13ed7d80c7e005088.html
## 2、重装Ubuntu16.04
下载地址：	
http://mirrors.aliyun.com/ubuntu-releases/16.04/ubuntu-16.04.5-desktop-amd64.iso
参考博客：https://blog.csdn.net/weixin_38233274/article/details/80237572

（1）	将ubuntu-16.04.4-desktop-amd64.iso放到C盘根目录，镜像文件里面有个casper文件夹，将文件vmlinuz 、initrd也拷贝到C盘根目录下。

（2）	运行EasyBCD，“添加新条目”->“NeoGrub”->“安装”。

（3）	配置->编辑menu.lst文件

（4）	title Install Ubuntu

root (hd0,0)
kernel (hd0,0)/vmlinuz boot=casper iso-scan/filename=/ubuntu-16.04.2-desktop-amd64.iso ro quiet splash locale=zh_CN.UTF-8
initrd (hd0,0)/initrd

（5）	重启（选择NeoGrub）

（6）	在安装之前打开终端Ctrl+Alt+T，输入sudo umount -l /isodevice，注意空格，可多执行一次，以确保将挂载的镜像移除，否则将无法进行安装。

（7）	您已安装的多个操作系统->其他选项

（8）	运行ubuntu安装程序安装Ubuntu16.04 LTS，交换空间一般跟内存条大小差不多就可以了，/和/home平分各100G差不多，最下面的挂载选在/所在的分区，当Windows系统重装时，就不会影响Ubuntu系统了

（9）	安装完成后重启直接进入Windows，运行EasyBCD，“添加新条目”->“NeoGrub”->“删除”，删除ubuntu的安装引导。

（10）	EasyBCD，“添加新条目”->“Linux/BSD”。类型选择 Grub2，名称可自定，驱动器选择/所在的分区。点击“添加条目”即可。

（11）	重启即可。删除安装引导选项。EasyBCD软件，进入一开始配置文件的那个位置，点击 remove 即可 ，重新启动就不会有引导安装的选项了。

## 3、配置固定IP

（1）	windows系统下查看自己的IP

（2）	Ubuntu下进行网络设置
   
## 4、更新源（如果我们的16.04内网源好使的了的话）
教程参考192.168.2.68/ubuntu/mannual.html(如果连不上就是不听话没有配置固定IP)

（1）	cd (sources.list位置)

（2）	sudo cp sources.list /etc/apt/sources.list

（3）	sudo apt-get update

## 5、安装NVIDIA显卡驱动
下载地址：https://www.nvidia.cn/Download/index.aspx?lang=cn	

参考博客：https://blog.csdn.net/xx_katherine/article/details/77754179

（1）	卸载原有驱动sudo apt-get purge nvidia*

（2）	禁用nouveau，创建blacklist-nouveau.conf

```
sudo vim /etc/modprobe.d/blacklist-nouveau.conf
```

编辑内容为：

```
blacklist nouveau
options nouveau modeset=0
```

（3）更新后重启系统

```
sudo update-initramfs –u
```

（4）关闭图形化界面
```
sudo service lightdm stop
```

（5）ctrl+alt+f1进入tty1命令行模式安装驱动
```
cd （驱动位置）
sudo sh ./NVIDIA*.run
```
（6）安装完成后重启图像化界面
```
sudo service lightdm start
```
（7）验证NVIDIA安装成功，成功打印出显卡信息

```
nvidia-smi

```
6、安装CUDA9.0

首先我要说一说为什么要安装9.0：

https://stackoverflow.com/questions/50442076/install-gpu-version-tensorflow-with-older-version-cuda-and-cudnn
 
> 历史经验告诉我们，我们实验需要TensorFlow-GPU>1.7.0，这就需要CUDA9.0+CUDNN7.0以上的配置（要对应）；而cuda9.0没有Ubuntu14的版本。如果你安装的是Ubuntu14.04或者其他低于Ubuntu16.04的版本，然后发现你要使用TensorFlow-GPU1.7.0以上版本的功能，那就可以休息一天，重新在装一遍，这就是为什么有此一文。

下载地址：

https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal

参考博客：https://blog.csdn.net/qlulibin/article/details/78714596

（1）	关闭图形化界面，ctrl+alt+f1进入tty1命令行模式安装驱动

（2）	进入run文件位置，执行如下命令，一直回车看完文档
```
sudo sh cuda_9.0.176_384.81_linux.run
```

（3）	根据提示输入，默认路径即可

（4）	进入图形化界面配置环境变量，运行如下命令打开profile文件
```
sudo gedit  /etc/profile
```

（5）	打开文件后在文件末尾添加路径，也就是安装目录，命令如下：（如果重启后报错，把这两句命令放在.bashrc中，参见cudnn安装报错解决办法）
```
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}} 
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

（6）	保存，然后重启电脑
```
sudo reboot
```

（7）	测试CUDA的Samples例子
```
cd  /usr/local/cuda-9.0/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

（8）	PASS：成功

（9）	安装补丁

## 7、安装Cudnn

下载地址：https://developer.nvidia.com/rdp/cudnn-download

参考博客：https://www.jianshu.com/p/69a10d0a24b9

验证cudnn正确安装：

https://blog.csdn.net/u014561933/article/details/79968539#4%E9%AA%8C%E8%AF%81

报错：参考博客：https://blog.csdn.net/mumodm/article/details/79502848
（1）	根据如下命令
```
cd ~
 sudo tar xvf cudnn-8.0-linux-x64-v5.1.tgz
 cd cuda/include
 sudo cp *.h /usr/local/include/
 cd ../lib64
 sudo cp lib* /usr/local/lib/
 cd /usr/local/lib# sudo chmod +r libcudnn.so.5.1.5
 sudo ln -sf libcudnn.so.7.2.1 libcudnn.so.7
 sudo ln -sf libcudnn.so.7 libcudnn.so
 sudo ldconfig
 ```
 
（2）验证是否正确安装

验证包：http://og9m6v6ow.bkt.clouddn.com/cudnn_samples_v7.tar.gz

解压到可写的文件夹下，进入
```
cd  cudnn_samples_v7/mnistCUDNN
```
（3）编译

```
make clean && make
```

（4）运行mnistCUDNN样例
```
 ./mnistCUDNN
 ```
 
（5）如果输出：Test passed!说明安装完成

（6）如果过程中报错，大部分情况下是环境没有配好
```
Error: libcudart.so.9.0: cannot open shared object file: No such file or directory
// 或者
Error: libcusolver.so.9.0: cannot open shared object file: No such file or direcctory
// 或者
Error: libcublas.so.9.0: cannot open shared object file: No such file or directory
```
参考博客：https://blog.csdn.net/mumodm/article/details/79502848

①第一种可靠的解决方法：
```
cd ~
sudo vi .bashrc
// 下滑到文件末，添加以下内容
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
// 刷新.bashrc
source .bashrc
// 以上解法是对生成了软连接的情况；如果没有生成软连接，则把以上的cuda改为cuda-9.0
```

②如果添加好了环境，还是出现同样的报错，则可以尝试以下解法：
```
cd ~
sudo cp /usr/local/cuda-9.0/lib64/libcudart.so.8.0 /usr/local/lib/libcudart.so.9.0 && sudo ldconfig
cp /usr/local/cuda-9.0/lib64/libcublas.so.9.0 /usr/local/lib/libcublas.so.9.0 && sudo ldconfig
cp /usr/local/cuda-9.0/lib64/libcurand.so.9.0 /usr/local/lib/libcurand.so.9.0 && sudo ldconfig
```

报哪个错就改哪个

③一般情况下，以上两种解法可以搞定问题的；如果还是报错libcusolver.so.9.0不存在，下面是算是一种解法：
```
sudo ldconfig /usr/local/cuda/lib64
```
8、安装anaconda（自带python3.6）

下载地址：https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh

参考博客：https://blog.csdn.net/xiaerwoailuo/article/details/70054429

（1）在命令行用python和python3命令查看python版本

Ubuntu16自带的是python2.7和python3.6，安装的

（2）进入Anaconda3-5.2.0-Linux-x86_64.sh文件位置，然后执行
```
bash Anaconda3-5.2.0-Linux-x86_64.sh
```
（3）一路回车/yes，会自动配置好环境变量，重启终端才会生效。重启后输入python，提示python 3.6.5 anaconda……说明安装完成

（4）通过import scipy验证是否安装成功

9、安装TensorFlow-GPU

下载地址：https://pypi.org/project/tensorflow-gpu/#files

参考博客：https://blog.csdn.net/taoqick/article/details/79171199

（1）进入文件路径
```
pip install tensorflow_gpu-1.10.1-cp36-cp36m-manylinux1_x86_64.whl
```
（2）安装过程中会报错，是因为离线安装缺少依赖包，踩过坑的会把包留着（这就是为什么上一步要安装anaconda，一方面anaconda方便管理python版本，另一方面就是会自动安装很多包，所以这一步也就几个文件需要自己手动安装），但系统不会自动安装压缩文件，pip install ……重复直到TensorFlow-GPU安装成功即可

（3）验证是否安装成功
```
python
>>>import tensorflow as tf
>>>a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
>>>b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
>>>c = tf.matmul(a, b)# Creates a session with log_device_placement set to True.
>>>sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))# Runs the op.
>>>print(sess.run(c))
```
（4）输出结果证明安装成功


## 10、安装pycharm
下载地址：
https://download.jetbrains.8686c.com/python/pycharm-community-2018.2.3.tar.gz

参考博客：https://blog.csdn.net/qq_38786209/article/details/78309191?readlog
https://blog.csdn.net/sinat_35257860/article/details/72737399

（1）进入文件路径
tar -xvzf pycharm-community-2018.2.3.tar.gz

（2）进入解压路径，运行

cd （解压文件路径）pycharm-community-2018.2.3/bin
sh pycharm.sh

（3）Pycharm启动方法：

参考博客：https://blog.csdn.net/sinat_35257860/article/details/72737399

a)sh pycharm.sh

b)https://blog.csdn.net/tmosk/article/details/72852330
```
cd /usr/share/applications/
sudo vim Pycharm.desktop
```
这里必须得用root权限sudo才能写入，然后在文件中写入以下内容。

```
 [Desktop Entry]
Type = Application     
Name = Pycharm
GenericName = Pycharm
Comment = Pycharm:The Python IDE
Exec = sh /home/lxq/Downloads/pycharm/bin/pycharm.sh
Icon = /home/lxq/Downloads/pycharm/bin/pycharm.png
Terminal = pycharm
Categories = Pycharm;
```

c)在pycharm工具里选择创建图标：Tools -> create desktop entry...（亲测这个最方便）

（4）配置编译环境file->settings->小齿轮->add->选择/usr/local/anaconda3bin/python3.6（总之是python3.6,选择所有项目都使用这个编译器。因为TensorFlow是这个版本的，没有他用其他编译器也可以）

（5）新建文件测试，成没成功你知道的

```
import tensorflow as tf
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))# Runs the op.
print(sess.run(c))
```
11、Anaconda下安装python版本的opencv-python和opencv-contrib-python

下载地址：https://pypi.org/project/opencv-python/
https://pypi.org/project/opencv-contrib-python/

参考博客：同上
进入这两个文件位置，在终端中输入两句命令：
```
pip install opencv_python-3.4.3.18-cp36-cp36m-manylinux1_x86_64.whl
pip install opencv_contrib_python-3.4.3.18-cp36-cp36m-manylinux1_x86_64.whl
```
OK，测试一下import cv2，成功

12、安装pytorch

下载地址：
Pytorch：https://pytorch.org/previous-versions/

torchvision 0.2.1：https://pypi.org/project/torchvision/#files

参考博客：https://blog.csdn.net/red_stone1/article/details/78727096

（1）进入PyTorch的下载目录，使用pip命令安装：

```
pip install torch-0.4.0-cp36-cp36m-linux_x86_64.whl
```
（2）在pypi下载，然后安装torchvision，可直接使用pip命令安装：
```
pip install torchvision
```
（3）测试，进入python环境
```
import torch
import torchvision
print(torch.cuda.is_available())#输出true
exit()
```
13、安装clion

下载地址：https://download.jetbrains.8686c.com/cpp/CLion-2018.2.3.tar.gz

参考博客：https://blog.csdn.net/u010925447/article/details/73251780

（1）	tar -zxvf CLion-2016.2.2.tar.gz

（2）	cd clion-2016.2.2/bin/  

（3）	./clion.sh  

（4）	验证码http://idea.lanyus.com/

（5）	注意新建工程测试的时候要把对应的CMakeList.txt中cmake的版本改成自己的！

14、安装QT

下载地址：https://pan.baidu.com/s/1o7H1y2I

参考博客：https://blog.csdn.net/lql0716/article/details/54564721

（1）将下载的安装文件qt-opensource-linux-x64-5.7.1.run拷贝到home/用户目录，如/home/user
（2）如果qt-opensource-linux-x64-5.7.1.run的属性中拥有者没有运行权限，则可用chmod命令添加执行权限:
（3）chmod u+x qt-opensource-linux-x64-5.7.1.run
（4）在终端执行：
```
./ qt-opensource-linux-x64-5.7.1.run
```
（5）跳出安装界面，一直点击下一步，直到安装完成即可。

（6）测试控制台程序
```
#include <QCoreApplication>
#include <stdio.h>
#include <iostream>
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    std::cout<<”hello”<<std::endl;#Ubuntu下printf不好使哦
    return a.exec();
}
```

15、安装opencv3.3和opencv_contrib

下载地址：https://github.com/opencv/opencv/archive/3.3.1.zip

https://github.com/opencv/opencv_contrib/archive/3.3.1.zip

参考博客：https://www.cnblogs.com/arkenstone/p/6490017.html

https://blog.csdn.net/xiangxianghehe/article/details/78780269

（1）安装依赖包
```
sudo apt-get install build-essential  
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev  
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev  
sudo apt-get install build-essential qt5-default ccache libv4l-dev libavresample-dev  libgphoto2-dev libopenblas-base libopenblas-dev doxygen  openjdk-8-jdk pylint libvtk6-dev
sudo apt-get install pkg-config
```
（2）解压下载好的包：

```
unzip opencv-3.3.1.zip
unzip opencv_contrib-3.3.1.zip
```
（3）解压完后需要将opencv_contrib.zip提取到opencv目录下，同时在该目录下新建一个文件夹build：
```
cp -r opencv_contrib-3.3.1 opencv-3.3.1  #复制opencv_contrib到opencv目录下
cd opencv-3.3.1
mkdir build   #新建文件夹build
```
（4）	进入build目录，并且执行cmake生成makefile文件：
```
cd build  
```
（5）	
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=/home/elsie/OPENCV/opencv-3.3.1/opencv_contrib-3.3.1/modules -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D CUDA_ARCH_BIN="6.1" -D CUDA_ARCH_PTX="" -D CUDA_FAST_MATH=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON -D BUILD_EXAMPLES=ON ..
```
注意：①CUDA_ARCH_BIN="6.1”这个需要去官网确认使用的GPU所对应的版本[查看这里](https://developer.nvidia.com/cuda-gpus)

②如果qt未安装可以删去此行;若因为未正确安装qt导致的Qt5Gui报错，可将build内文件全部删除后重新cmake，具体可以参考[这里](http://stackoverflow.com/questions/17420739/opencv-2-4-5-and-qt5-error-s)

③OPENCV_EXTRA_MODULES_PATH就是你 opencv_contrib-3.3.1下面的modules目录，请按照自己的实际目录修改地址。

④后面的两点不可省略 

（6）生成完毕提示：（没有错误！有坑！）
```
--   Install path:                  /usr/local
-- 
--   cvconfig.h is in:        /home/elsie/OPENCV/opencv-3.3.1/opencv_contrib-3.3.1 /build
-- -----------------------------------------------------------------
-- 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/elsie/OPENCV/opencv-3.3.1/opencv_contrib-3.3.1/modules /build
```
注意：虽然Configuring done  -- Generating done这里仍然会有几个坑影响后面的make

1 过程中需要下载诸如ippicv_2017u3_lnx_intel64_20170822.tgz的东西（在cmake的输出中往上拉，一般都会失败），如果下载失败：
下载地址：https://github.com/opencv/opencv_3rdparty/branches/all
下载的东西名叫opencv_3rdparty-ippicv-master_20170822.zip，解压找到ippicv_2017u3_lnx_intel64_general_20170822.tgz文件，拷贝到某目录，然后把~/opencv-3.3.1/3rdparty/ippicv中的ippicv.cmake文件中的GitHub下载地址修改为自己的本地地址。
注意：网页中说的是修改为files：//地址，不需要files：//,这是从服务器下载，路径直接写文件路径即可。

2 缺少boostdesc_bgm.i boostdesc_bgm_bi.i boostdesc_bgm_hd.i boostdesc_binboost_064.i boostdesc_binboost_128.i boostdesc_binboost_256.i boostdesc_lbgm.i vgg_generated_120.i vgg_generated_48.i vgg_generated_64.i vgg_generated_80.i等文件。

下载地址：https://download.csdn.net/download/sinat_39805237/10563950
所有文件放到opencv_contrib-3.3.1/modules/xfeatures2d/src中
然后把opencv_contrib-3.3.1/modules/xfeatures2d/cmake文件夹里的download_boostdesc.cmake 和download_vgg.cmake中下载地址那一部分改成……/src那一段。

3 es10_300x300_ssd_iter_140000.caffemodel和tiny-dnn下的v1.0.0a3.tar.gz找不到

下载地址：https://download.csdn.net/download/u010782463/10309793
https://download.csdn.net/download/wjskeepmaking/9824941?web=web
解决方法同①，修改~/ opencv-3.3.1/ opencv_contrib-3.3.1/modules/dnn_modern/cmake里的cmakelist.txt改成本地路径

（7）	在cmake成功之后，就可以在build文件下make了：
```
make -j8        #8线程编译
make install
```
（8）	测试
```
/**
* @概述：采用FAST算子检测特征点，采用SIFT算子对特征点进行特征提取，并使用BruteForce匹配法进行特征点的匹配
* @类和函数：FastFeatureDetector + SiftDescriptorExtractor + BruteForceMatcher
*/


#include<opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv)
{
    Mat objImage = imread("1.jpg", IMREAD_COLOR);
    Mat sceneImage = imread("2.jpg", IMREAD_COLOR);
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> obj_keypoint, scene_keypoint;
    detector->detect(objImage, obj_keypoint);
    detector->detect(sceneImage, scene_keypoint);
    //computer the descriptors
    Mat obj_descriptors, scene_descriptors;
    detector->compute(objImage, obj_keypoint, obj_descriptors);
    detector->compute(sceneImage, scene_keypoint, scene_descriptors);
    //use BruteForce to match,and get good_matches
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(obj_descriptors, scene_descriptors, matches);
    sort(matches.begin(), matches.end());  //筛选匹配点
    vector<DMatch> good_matches;
    for (int i = 0; i < min(50, (int)(matches.size()*0.15)); i++) {
        good_matches.push_back(matches[i]);
    }
    //draw matches
    Mat imgMatches;
    drawMatches(objImage, obj_keypoint, sceneImage, scene_keypoint,good_matches, imgMatches);
    //get obj bounding
    vector<Point2f> obj_good_keypoint;
    vector<Point2f> scene_good_keypoint;
    for (int i = 0; i < good_matches.size(); i++) {
        obj_good_keypoint.push_back(obj_keypoint[good_matches[i].queryIdx].pt);
        scene_good_keypoint.push_back(scene_keypoint[good_matches[i].trainIdx].pt);
    }
    vector<Point2f> obj_box(4);
    vector<Point2f> scene_box(4);
    obj_box[0] = Point(0, 0);
    obj_box[1] = Point(objImage.cols, 0);
    obj_box[2] = Point(objImage.cols, objImage.rows);
    obj_box[3] = Point(0, objImage.rows);
    Mat H = findHomography(obj_good_keypoint, scene_good_keypoint, RANSAC); //find the perspective transformation between the source and the destination
    perspectiveTransform(obj_box, scene_box, H);
    line(imgMatches, scene_box[0]+Point2f((float)objImage.cols, 0), scene_box[1] + Point2f((float)objImage.cols, 0), Scalar(0, 255, 0), 2);
    line(imgMatches, scene_box[1] + Point2f((float)objImage.cols, 0), scene_box[2] + Point2f((float)objImage.cols, 0), Scalar(0, 255, 0), 2);
    line(imgMatches, scene_box[2] + Point2f((float)objImage.cols, 0), scene_box[3] + Point2f((float)objImage.cols, 0), Scalar(0, 255, 0), 2);
    line(imgMatches, scene_box[3] + Point2f((float)objImage.cols, 0), scene_box[0] + Point2f((float)objImage.cols, 0), Scalar(0, 255, 0), 2);
    //show the result                                                                   
    imshow("匹配图", imgMatches);
    //save picture file
    imwrite("final.jpg",imgMatches);
    waitKey(0);
    return 0;
}
```

项目的cmakelist.txt配置如下：
```
cmake_minimum_required(VERSION 3.5)
project(untitled)
set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES main.cpp)
add_executable(untitled main.cpp)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES main.cpp)
target_link_libraries(untitled ${OpenCV_LIBS})
```
（9）	链接库共享。编译安装完毕之后，为了让你的链接库被系统共享，让编译器发现，需要执行管理命令ldconfig：
```
sudo ldconfig -v  
```

16、	【这是一段失败的旅程，后来我放弃了，不过可以解决的…】安装OPENCV2.4.9

下载地址：http://jaist.dl.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.9/opencv-2.4.9.zip
	参考博客：（参考的有点多，主要都列在排错上了）
	
https://blog.csdn.net/u014527548/article/details/80251046

（1）	解压到任意目录，进入源码目录
```
unzip opencv-2.4.9.zip
cd opencv-2.4.9
```

（2）	安装下列依赖

```
sudo apt-get install build-essential cmake libgtk2.0-dev pkg-config python-dev python-numpy libavcodec-dev libavformat-dev libswscale-dev

sudo apt-get install build-essential libgtk2.0-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen3-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev 
default-jdk ant libvtk5-qt4-dev
```

注意：这里可能会报错：
```
libgtk2.0-dev : 依赖: libgtk2.0-0 (= 2.24.23-0ubuntu1) 但是 2.24.23-0ubuntu1.1 正要被安装
                 依赖: libpango1.0-dev (>= 1.20) 但是它将不会被安装
                 依赖: libcairo2-dev (>= 1.6.4-6.1) 但是它将不会被安装
                 推荐: debhelper 但是它将不会被安装
E: 无法修正错误，因为您要求某些软件包保持现状，就是它们破坏了软件包间的依赖关系。
```

如果忽略了这个错误继续安装，后面的OpenCV可能不能正常使用，我们要解决这个问题。

方法：
```
sudo aptitude install libgtk2.0-dev
```
下面会出来一堆解决方案，都是保留……，然后问是否接受这个解决方案。

这时候要选No！因为出现这个问题的根本原因是安装包A依赖于C的旧版本,而机器上已经存在了C的新版本,此新的版本又是B的依赖,所以就会出现版本的依赖混乱问题。
直到出现了“降级”这样的解决方案，yes。降级完之后重新安装即可。其他类似问题同样可以参考。

也有可能是源的问题，不过，离线安装既然做不到在线更新源，那就酱紫继续吧。

还有，可能会报python-numpy的包依赖错误。这里我是先装好了TensorFlow之前一套，才想起来更新源的。不知道是不是这个原因导致的依赖问题，python-numpy降级以后，记得先测试一下import tensorflow as tf能否正常工作，如果不行的话，再测cuda\cudnn能否运行示例程序。

我这里是可以运行示例程序，但找不到TensorFlow，现在需要重装TensorFlow。步骤如上……

然后保险起见再执行一下
```
sudo apt-get install build-essential cmake libgtk2.0-dev pkg-config python-dev python-numpy libavcodec-dev libavformat-dev libswscale-dev
```
（3）进入cmake
```
cd cmake
```
（4）	cmake编译生成Makefile，安装所有的lib文件都会被安装到/usr/local目录
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..  
```
不报错的人生一点都不完美
```
CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
Please set them or make sure they are set and tested correctly in the CMake files:
CUDA_nppi_LIBRARY (ADVANCED)
linked by target "opencv_cudev" in directory D:/Cproject/opencv/opencv/sources/modules/cudev
linked by target "opencv_cudev" in directory D:/Cproject/opencv/opencv/sources/modules/cudev
linked by target "opencv_test_cudev" in directory D:/Cproject/opencv/opencv/sources/modules/cudev/test
linked by target "opencv_core" in directory D:/Cproject/opencv/opencv/sources/modules/core
linked by target "opencv_core" in directory D:/Cproject/opencv/opencv/sources/modules/core
linked by target "opencv_test_core" in directory D:/Cproject/opencv/opencv/sources/modules/core
linked by target "opencv_perf_core" in directory D:/Cproject/opencv/opencv/sources/modules/core
……
Please set them or make sure they are set and tested correctly in the CMake files:
CUDA_nppi_LIBRARY (ADVANCED)
linked by target "opencv_cudev" in directory D:/Cproject/opencv/opencv/sources/modules/cudev
linked by target "opencv_cudev" in directory D:/Cproject/opencv/opencv/sources/modules/cudev
linked by target "opencv_test_cudev" in directory D:/Cproject/opencv/opencv/sources/modules/cudev/test
linked by target "opencv_core" in directory D:/Cproject/opencv/opencv/sources/modules/core
linked by target "opencv_core" in directory D:/Cproject/opencv/opencv/sources/modules/core
linked by target "opencv_test_core" in directory D:/Cproject/opencv/opencv/sources/modules/core
linked by target "opencv_perf_core" in directory D:/Cproject/opencv/opencv/sources/modules/core
linked by target "opencv_test_cudaarithm" in directory 

……
```

为啥呢？满屏都是错

大神告诉我们，原因是cuda9.0不再支持2.0架构

参考博客：https://blog.csdn.net/u014613745/article/details/78310916
https://blog.csdn.net/mystylee/article/details/79035585

https://stackoverflow.com/questions/46584000/cmake-error-variables-are-set-to-notfound

解决方案抄录如下：（注意OpenCV2版本的不执行https://blog.csdn.net/u014613745/article/details/78310916中的第四步，否则会报错；OpenCV3架构的需要执行，否则会报错）

4	找到FindCUDA.cmake文件（opencv-2.4.9下cmake目录），找到行find_cuda_helper_libs(nppi)修改为：
```
find_cuda_helper_libs(nppial)
find_cuda_helper_libs(nppicc)
find_cuda_helper_libs(nppicom)
  find_cuda_helper_libs(nppidei)
  find_cuda_helper_libs(nppif)
  find_cuda_helper_libs(nppig)
  find_cuda_helper_libs(nppim)
  find_cuda_helper_libs(nppist)
find_cuda_helper_libs(nppisu)
 find_cuda_helper_libs(nppitc)
 ```
5	找到行
```
set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppi_LIBRARY};${CUDA_npps_LIBRARY}")修改为
set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppial_LIBRARY};${CUDA_nppicc_LIBRARY};${CUDA_nppicom_LIBRARY};${CUDA_nppidei_LIBRARY};${CUDA_nppif_LIBRARY};${CUDA_nppig_LIBRARY};${CUDA_nppim_LIBRARY};${CUDA_nppist_LIBRARY};${CUDA_nppisu_LIBRARY};${CUDA_nppitc_LIBRARY};${CUDA_npps_LIBRARY}")
```
6	找到行
```
unset(CUDA_nppi_LIBRARY CACHE)修改为
unset(CUDA_nppial_LIBRARY CACHE)
unset(CUDA_nppicc_LIBRARY CACHE)
unset(CUDA_nppicom_LIBRARY CACHE)
unset(CUDA_nppidei_LIBRARY CACHE)
unset(CUDA_nppif_LIBRARY CACHE)
unset(CUDA_nppig_LIBRARY CACHE)
unset(CUDA_nppim_LIBRARY CACHE)
unset(CUDA_nppist_LIBRARY CACHE)
unset(CUDA_nppisu_LIBRARY CACHE)
unset(CUDA_nppitc_LIBRARY CACHE)
```
7	cuda9中有一个单独的halffloat(cuda_fp16.h)头文件,也应该被包括在opencv的目录里,将头文件cuda_fp16.h添加至
```opencv\modules\gpu\include\opencv2\gpu\common.hpp```
即在common.hpp中添加
```
#include <cuda_fp16.h>
```
8	重新执行
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..  
```
（5）	继续OpenCV的安装，在cmake文件夹下执行以下命令
```
sudo make install 
```
（6）	还是会报错，不报错的人生不完美。这次的错误长这样：
```
nvcc fatal   : Unsupported gpu architecture 'compute_11'
CMake Error at cuda_compile_generated_matrix_operations.cu.o.cmake:208 (message):
Error generating/home/elsie/Documents/opencv2.4.9/build/modules/core/CMakeFiles/cuda_compile.dir/__/dynamicuda/src/cuda/./cuda_compile_generated_matrix_operations.cu.o
make[2]: ***
[modules/core/CMakeFiles/cuda_compile.dir/__/dynamicuda/src/cuda/./cuda_compile_generated_matrix_operations.cu.o] Error 1
make[1]: *** [modules/core/CMakeFiles/opencv_core.dir/all] Error 2 make[1]: *** Waiting for unfinished jobs.…
```
解决一下吧（虽然只写了五个字，可是我卡了半天）：
```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D CUDA_GENERATION=Kepler ..
```
然后就按照上面的解决办法把丢失的文件补进去就好了。


作者：徐樱笑
