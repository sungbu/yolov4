from models import Yolov4
import cv2
import numpy as np
# 加载待预测的图像
image_path = "/Users/mr.happy/Documents/数据集训练/coco2014/dataset/img/COCO_train2014_000000578793.jpg" # 0
# image_path = "/Users/mr.happy/Documents/数据集训练/coco2014/dataset/img/COCO_train2014_000000580446.jpg" # 1
# image_path = "/Users/mr.happy/Documents/数据集训练/coco2014/dataset/img/COCO_train2014_000000581921.jpg" # 2
# image_path = "/Users/mr.happy/Documents/数据集训练/coco2014/img/test.jpg"  #3
class_name_path = '/Users/mr.happy/Documents/数据集训练/coco2014/dataset/coco.txt'
# raw_img = cv2.imread(image_path)[:, :, ::-1]
# raw_img = cv2.resize(raw_img, (416,416))
# raw_img = raw_img / 255.
# imgs = np.expand_dims(raw_img, axis=0)

model = Yolov4(weight_path=None, 
               class_name_path=class_name_path)
inference_model = model.load_model("/Users/mr.happy/Documents/数据集训练/coco2014/H51/H5-epoch0/model.h5")

# print(inference_model(imgs))

print(model.predict(image_path))