#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import tensorflow.keras.backend as K
import tensorflow as tf


def xywh_to_x1y1x2y2(boxes):
    return tf.concat([boxes[..., :2] - boxes[..., 2:] * 0.5, boxes[..., :2] + boxes[..., 2:] * 0.5], axis=-1)


# x,y,w,h
def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # w * h
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # (x, y, w, h) -> (x0, y0, x1, y1)
    boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes2 = xywh_to_x1y1x2y2(boxes2)

    # coordinates of intersection
    top_left = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection_xy = tf.maximum(bottom_right - top_left, 0.0)

    intersection_area = intersection_xy[..., 0] * intersection_xy[..., 1]
    union_area = boxes1_area + boxes2_area - intersection_area

    return 1.0 * intersection_area / (union_area + tf.keras.backend.epsilon())


def bbox_giou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # w*h
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # (x, y, w, h) -> (x0, y0, x1, y1)
    boxes1 = xywh_to_x1y1x2y2(boxes1)
    boxes2 = xywh_to_x1y1x2y2(boxes2)

    top_left = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    intersection_xy = tf.maximum(bottom_right - top_left, 0.0)
    intersection_area = intersection_xy[..., 0] * intersection_xy[..., 1]

    union_area = boxes1_area + boxes2_area - intersection_area

    iou = 1.0 * intersection_area / (union_area + tf.keras.backend.epsilon())

    enclose_top_left = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_bottom_right = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])

    enclose_xy = enclose_bottom_right - enclose_top_left
    enclose_area = enclose_xy[..., 0] * enclose_xy[..., 1]

    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)

    return giou


def bbox_ciou(boxes1, boxes2):
    '''
    ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:
    '''
    boxes1_x0y0x1y1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    boxes1_x0y0x1y1 = tf.concat([tf.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
    boxes2_x0y0x1y1 = tf.concat([tf.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                                 tf.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

    # area
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # top-left and bottom-right coord, shape: (8, 13, 13, 3, 2)
    left_up = tf.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = tf.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # intersection area and iou
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # top-left and bottom-right coord of the enclosing rectangle, shape: (8, 13, 13, 3, 2)
    enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # diagnal ** 2
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = K.pow(enclose_wh[..., 0], 2) + K.pow(enclose_wh[..., 1], 2)

    # center distances between two rectangles
    p2 = K.pow(boxes1[..., 0] - boxes2[..., 0], 2) + K.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # add av
    atan1 = tf.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-9))
    atan2 = tf.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-9))
    v = 4.0 * K.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou

# args：这是函数的输入参数列表。根据函数内部的使用情况，它似乎包含以下元素：
# args[0]：这对应于最小尺度检测层的输出张量（conv_sbbox）。其形状可能是 (?, ?, ?, 3*(num_classes+5))，其中 '?' 表示可变维度。
# args[1]：这对应于中间尺度检测层的输出张量（conv_mbbox）。形状类似于 conv_sbbox。
# args[2]：这对应于最大尺度检测层的输出张量（conv_lbbox）。形状类似于 conv_sbbox。
# args[3]：label_sbbox：这是最小尺度检测层的标签张量。它的形状是 (?, ?, ?, 3, num_classes+5)。在这个形状中，第一个维度表示批量大小（batch size），接下来的三个维度可能对应于图像的高度、宽度和通道数，然后是每个网格单元（grid cell）预测的框的数量（通常是3），最后一个维度是目标类别数加上5（这5个值通常包括目标的位置（x、y、w、h）以及目标存在的置信度）。
# args[4]：label_mbbox：这是中等尺度检测层的标签张量，形状类似于 label_sbbox。
# args[5]：label_lbbox：这是最大尺度检测层的标签张量，形状类似于 label_sbbox。
# args[6]：true_bboxes：这是真实边界框（ground truth bounding boxes）的张量。它的形状是 (?, 50, 4)，其中第一个维度表示批量大小，第二个维度是最大允许的真实边界框数量（通常为50），最后一个维度是边界框的坐标信息（通常是边界框的左上角和右下角的坐标）。
#num_classes：目标类别的数量。在目标检测任务中，通常是要检测的物体类别数目。
# iou_loss_thresh：IOU（交并比）损失的阈值。IOU 是衡量检测框与真实目标框之间重叠程度的指标，该阈值可能用于判断是否计算 CIoU 损失。

# anchors：这是一个包含锚框信息的列表。在 YOLO 中，锚框用于辅助预测目标框的位置和大小。通常，锚框是在训练数据集上通过聚类等方法得到的一组预定义框。
def yolo_loss(args, num_classes, iou_loss_thresh, anchors):
    conv_lbbox = args[2]   # (?, ?, ?, 3*(num_classes+5))
    conv_mbbox = args[1]   # (?, ?, ?, 3*(num_classes+5))
    conv_sbbox = args[0]   # (?, ?, ?, 3*(num_classes+5))
    label_sbbox = args[3]   # (?, ?, ?, 3, num_classes+5)
    label_mbbox = args[4]   # (?, ?, ?, 3, num_classes+5)
    label_lbbox = args[5]   # (?, ?, ?, 3, num_classes+5)
    true_bboxes = args[6]   # (?, 50, 4)
    pred_sbbox = decode(conv_sbbox, anchors[0], 8, num_classes)
    pred_mbbox = decode(conv_mbbox, anchors[1], 16, num_classes)
    pred_lbbox = decode(conv_lbbox, anchors[2], 32, num_classes)
    sbbox_ciou_loss, sbbox_conf_loss, sbbox_prob_loss = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_bboxes, 8, num_classes, iou_loss_thresh)
    mbbox_ciou_loss, mbbox_conf_loss, mbbox_prob_loss = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_bboxes, 16, num_classes, iou_loss_thresh)
    lbbox_ciou_loss, lbbox_conf_loss, lbbox_prob_loss = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_bboxes, 32, num_classes, iou_loss_thresh)


# 在YOLO损失函数中，将不同尺度的边界框位置损失（Bounding Box Regression Loss）加权求和后，使用了一个倍数进行调整。这个倍数通常是通过实验调整得到的，目的是平衡不同部分损失函数的重要性，以便更好地指导模型的训练。

    # 衡量预测边界框与真实边界框之间的位置差异
    ciou_loss = (lbbox_ciou_loss + sbbox_ciou_loss + mbbox_ciou_loss) * 3.54
    # 置信度损失衡量了模型对目标存在的置信度预测与实际情况之间的误差 置信度损失可以被视为一个二元分类问题，其目标是预测每个边界框内是否包含目标
    conf_loss = (lbbox_conf_loss + sbbox_conf_loss + mbbox_conf_loss) * 64.3
    # 类别概率损失用于衡量模型对目标类别的预测准确性，同样采用交叉熵损失函数计算
    # prob_loss 是类别概率损失的总和，由不同尺度（large、medium、small）的边界框的类别概率损失之和组成。
    prob_loss = (lbbox_prob_loss + sbbox_prob_loss + mbbox_prob_loss) * 1

    #总损失函数
    return ciou_loss+conf_loss+prob_loss


def loss_layer(conv, pred, label, bboxes, stride, num_class, iou_loss_thresh):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size,
                             3, 5 + num_class))
    conv_raw_prob = conv[:, :, :, :, 5:]
    conv_raw_conf = conv[:, :, :, :, 4:5]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    # Coordinate loss
    ciou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)  # (8, 13, 13, 3, 1)
    # ciou = tf.expand_dims(bbox_ciou(pred_xywh, label_xywh), axis=-1)  # (8, 13, 13, 3, 1)
    input_size = tf.cast(input_size, tf.float32)

    # loss weight of the gt bbox: 2-(gt area/img area)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)  # iou loss for respond bbox

    # Classification loss for respond bbox
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    expand_pred_xywh = pred_xywh[:, :, :, :, np.newaxis, :]  # (?, grid_h, grid_w, 3, 1, 4)
    expand_bboxes = bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]  # (?, 1, 1, 1, 70, 4)
    iou = bbox_iou(expand_pred_xywh, expand_bboxes)  # IoU between all pred bbox and all gt (?, grid_h, grid_w, 3, 70)
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)  # max iou: (?, grid_h, grid_w, 3, 1)

    # ignore the bbox which is not respond bbox and max iou < threshold
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_thresh, tf.float32)

    # Confidence loss
    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return ciou_loss, conf_loss, prob_loss


def decode(conv_output, anchors, stride, num_class):
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor_per_scale = len(anchors)
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_class))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

