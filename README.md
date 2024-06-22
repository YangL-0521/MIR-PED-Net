## Required environment
pytorch >= 1.7.1
The rest of the required libraries are in requirements.txt
     

## Training steps
1. Dataset preparation
 This article uses VOC format for training, you need to make your own dataset before training, **
Put the label file in Annotation under VOC2007 folder under VOCdevkit folder before training. (Tag file format is xml format)
Before training, put the image files in JPEGImages under VOC2007 folder under VOCdevkit folder.

2. Processing of the dataset
 After placing the dataset, we need to use voc_annotation.py to get 2007_train.txt and 2007_val.txt for training.
Change the arguments in voc_annotation.py. The first training can simply modify the classes_path, which is used to point to the txt corresponding to the detected class.
When training your own dataset, you can create your own cls_classes.txt with the classes you want to distinguish between.
The model_data/cls_classes.txt file contains:
```
```python
cat
dog
...
```
Change the classes_path in voc_annotation.py to match cls_classes.txt, and run voc_annotation.py.


3. Start train  
** There are many parameters for training, all in train.py, but the most important part is again classes_path in train.py. **
**classes_path is used to point to the txt corresponding to the detected class, which is the same txt as in voc_annotation.py! The dataset on which you train yourself will have to change! **
After changing classes_path, we can run train.py to start training, and after training for a number of epochs, the weights will be generated in the logs folder. 

4. Training result prediction  
We will use two files called yolo.py and predict.py to predict the training results. Change model_path and classes_path in yolo.py.
**model_path points to the trained weights file, in the logs folder.
classes_path points to the txt corresponding to the detected class. **
Once you've made these changes, you can run predict.py to test for them. After running, input the image path to detect.  

## Prediction steps
1. Follow the training steps。  
2.In yolo.py, modify model_path and classes_path to match our trained files in the following sections: **model_path corresponds to the weights file in the logs folder, and classes_path is the class corresponding to model_path **.
 
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolox_s.pth',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [640, 640],
    #---------------------------------------------------------------------#
    #   所使用的YoloX的版本。nano、tiny、s、m、l、x
    #---------------------------------------------------------------------#
    "phi"               : 's',
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : True,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```

3. run predict.py，input  
```python
img/street.jpg
```

4. Set it up in predict.py for fps testing and video detection. 

## Evaluation Steps

1. This paper uses VOC format for evaluation.
   
2. If you have run voc_annotation.py before training, the code will automatically split the dataset into training, validation, and test sets. If you want to change the test percentage, you can change trainval_percent in voc_annotation.py. trainval_percent specifies the ratio of (training + validation) to test set.Default (training + validation): test set = 9:1. train_percent specifies the ratio of the training set to the validation set (training set + validation set).The default is training set: validation set = 9:1.

3. After dividing the test set using voc_annotation.py, go to get_map.py and change the classes_path, which will point to the txt for the test class, which will be the same as the training txt. The data set on which you evaluate yourself will have to be modified.
   
4. Change model_path and classes_path in yolo.py. **model_path points to the trained weights file, in the logs folder. classes_path points to the txt corresponding to the detected class. **
Run get_map.py to get the evaluation results, which are stored in the map_out folder.

## Acquisition Method of Infrared Pedestrian Dataset
FLIR dataset download address is: https://www.flir.com/oem/adas/adas-dataset-form/. 
KAIST dataset download address is: https://github.com/SoonminHwang/rgbt-ped-detection.

NOTE: 
If a weight file is required, the author can be contacted
