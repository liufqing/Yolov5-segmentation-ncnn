# Yolov5 Segmentation with NCNN

> This branch is specifically made for ID card recognition and rotation.

![cmnd](./output/test.jpg)

---

## How to run

1. Put NCNN model (.bin and .param) to the "models" folder.
2. Put inference image to "input" folder
3. Put class names text file ( *.txt ) to "data" folder ( See the tree below )
4. Run yolov5-seg.exe in cmd ( assumed that the *.bin and *.param are both in models folder )

```
+---data
|       idcard.txt
|       
+---input
|       00.jpg
|       01.jpg
|       02.jpg
|       ...
|
+---models
|       yolov5s-seg-idcard-best.ncnn.bin
|       yolov5s-seg-idcard-best.ncnn.param
|       yolov5s-seg-idcard-2.ncnn.bin
|       yolov5s-seg-idcard-2.ncnn.param
|
+---output
|       00.jpg
|       01.jpg
|       02.jpg
|       ...
|
+---yoloncnn.exe
|
```  

```
Usage

yolov5-seg [option] <value>

Options
    --model <ncnn model name>           = Explicitly specify the ncnn model name. Default yolov5s-seg-idcard-2.ncnn
    --data <class names txt file>       = Explicitly specify the class names txt file. Default idcard.txt
    --source <input source>             = Explicitly specify the input source. Default input
    --output <output folder>            = Explicitly specify the output folder. Default output
    --size <target size>                = Specify the target size. Default 640
    --conf <confident threshold>        = Specify the confident threshold. Default 0.25
    --nms <nms threshold>               = Specify the nms threshold. Default 0.45
    --max-obj <max objects detection>   = Specify the max objects detection. Default 1
    --dynamic                           = Dynamic inference flag. Default false
    --agnostic                          = Agnostic nms flag. Default false
    --contour                           = Draw contour instead of mask flag. Default false
    --save                              = Save output flag. Default false
    --save--text                        = Save output label to a text file flag. Default false
    --crop                              = Crop output flag. Default false
    --rotate                            = Rotate output flag. Default false
```

**Note** :

source can be :

- image file path   
- video file path
- 0 for webcam
- folder path for run all images in input folder

For example :

```
yolov5-seg --source input/test.jpg --model yolov5s-seg-idcard-2.ncnn --save --rotate
```

```
4 argument(s) passed
model     = models\yolov5s-seg-idcard-2.ncnn.bin
param     = models\yolov5s-seg-idcard-2.ncnn.param
input     = input\test.jpg
data      = data\idcard.txt
size      = 640
conf      = 0.25
nms       = 0.45
maxObj    = 1
dynamic   = 0
contour   = 0
agnostic  = 0
crop      = 0
save      = 1
saveTxt   = 0
saveMask  = 0
rotate    = 1
------------------------------------------------
Inference time = 0.178 (seconds)
Objects count = 1
5 0.977652 37 87 594 591
Output saved at output\test.jpg
```

![test_0_cmt9_front.jpg](./output/rotate/test_0_cmt9_front.jpg)

---

## Wiki

- [Setup for windows, using visual studio 2022](https://github.com/canh25xp/Yolov5-segmentation-ncnn/wiki/How-to-build)
- [Convert pytorch to ncnn](https://github.com/canh25xp/Yolov5-segmentation-ncnn/wiki/Convert-pytorch-model-to-ncnn-model)
