from utils import DataGenerator, read_annotation_lines
from models import Yolov4
import tensorflow as tf
from keras.callbacks import ModelCheckpoint


train_lines, val_lines = read_annotation_lines('/Users/mr.happy/Documents/数据集训练/coco2014/dataset/label/train.txt', 
                                               test_size=0.1)
FOLDER_PATH = '/Users/mr.happy/Documents/数据集训练/coco2014/dataset/img'
class_name_path = '/Users/mr.happy/Documents/数据集训练/coco2014/dataset/coco.txt'

model_path = './run/yolov4.weights'
saveing_path = './saveing'
data_gen_train = DataGenerator(train_lines, 
                               class_name_path, 
                               FOLDER_PATH)
data_gen_val = DataGenerator(val_lines, 
                             class_name_path, 
                             FOLDER_PATH)

model = Yolov4(weight_path="/Users/mr.happy/Documents/数据集训练/coco2014/H51/H5-epoch0/model.h5", 
               class_name_path=class_name_path)

batch_data1,batch_data12 = data_gen_train.__getitem__(0)  # 获取第一个批次的数据
# print(batch_data1,batch_data12)

checkpoint = ModelCheckpoint(saveing_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


model.fit(data_gen_train, 
          initial_epoch=0,
          epochs=10000, 
          val_data_gen=data_gen_val,
          callbacks=[checkpoint])

model.save_model(model_path)