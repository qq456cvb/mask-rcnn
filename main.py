import tensorflow as tf
import generator
import ResNet_w_FPN
import RPN
import RoiAlign
import config
import rcnn_head
import tensorflow.contrib.slim as slim
import numpy as np
import functools
import utils
import cv2
import itertools


def rpn_model_fn(features, labels, mode):
    feature_maps = ResNet_w_FPN.forward(features['img'])
    pred_rpn_anchor_logits, pred_rpn_anchor_probs, pred_rpn_anchor_deltas = RPN.forward(feature_maps)
    class_loss, bbox_loss = tf.map_fn(RPN.rpn_loss, (pred_rpn_anchor_logits, pred_rpn_anchor_deltas, labels['rpn_labels'], labels['rpn_deltas'], labels['rpn_mask'], labels['rpn_positive_range'], labels['rpn_positive_mask']), dtype=(tf.float32, tf.float32))
    class_loss = tf.reduce_sum(class_loss)
    # shape = tf.shape(bbox_loss, name='shape')
    bbox_loss = tf.reduce_sum(bbox_loss)
    loss = tf.identity(class_loss + bbox_loss, name='loss')

    all_anchors = tf.convert_to_tensor(utils.generate_anchors(format=utils.BBOX_FORMAT.YXYX), dtype=tf.float32)
    all_anchors = tf.tile(tf.expand_dims(all_anchors, 0), [tf.shape(pred_rpn_anchor_logits)[0], 1, 1])
    proposals = tf.map_fn(RPN.inference, (pred_rpn_anchor_probs, pred_rpn_anchor_deltas, all_anchors), dtype=tf.float32)
    # get proposals in YXYX format
    proposals = tf.identity(proposals, name='proposals')
    rois, target_class, target_deltas, target_mask = tf.map_fn(utils.generate_mask_rcnn_x_y_tf, (proposals, labels['bboxs'],
                                                               labels['cls'],
                                                               labels['masks'],
                                                               labels['valid_label_ranges']),
                                                              dtype=(tf.float32, tf.int32, tf.float32, tf.float32))
    # rois = tf.Print(rois, [tf.shape(rois), tf.shape(target_class), tf.shape(target_deltas), tf.shape(target_mask)])
    mrcnn_cls_bbox_in = RoiAlign.forward(rois, feature_maps[:-1], config.CLS_BBOX_ROI_POOL_SIZE)
    mrcnn_mask_in = RoiAlign.forward(rois, feature_maps[:-1], config.MASK_ROI_POOL_SIZE)
    mrcnn_mask_out_logits, _ = rcnn_head.forward_mask(mrcnn_mask_in)
    mrcnn_cls_logits, _, mrcnn_deltas = rcnn_head.forward_cls_bbox(mrcnn_cls_bbox_in)
    cls_loss, bbox_loss, mask_loss = tf.map_fn(rcnn_head.mrcnn_loss,
                           (mrcnn_cls_logits, mrcnn_deltas, mrcnn_mask_out_logits, target_class, target_deltas, target_mask, rois),
                           dtype=(tf.float32, tf.float32, tf.float32))

    # return proposals, rois, target_class, target_mask, mrcnn_cls_bbox_in, mrcnn_deltas, cls_loss, bbox_loss, mask_loss

    global_step = tf.train.get_global_step()
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def train_input_fn():
    gen = generator.data_generator
    dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32),
                                             (
                                                 tf.TensorShape([None, None, 3]),
                                                 tf.TensorShape([None, 4]),
                                                 tf.TensorShape([config.RPN_ANCHORS_TRAIN_PER_IMAGE, 2]),
                                                 tf.TensorShape([None, 4]),
                                                 tf.TensorShape([None]),
                                                 tf.TensorShape([]),
                                                 tf.TensorShape([None]),
                                                 tf.TensorShape([None]),
                                                 tf.TensorShape([None, config.MASK_OUTPUT_SHAPE, config.MASK_OUTPUT_SHAPE]),
                                                 tf.TensorShape([])
                                             ))
    dataset = dataset.batch(2)
    # dataset = dataset.prefetch(1)
    next_batch = dataset.make_one_shot_iterator().get_next()
    return {
        'img': next_batch[0]
    }, {
        'bboxs': next_batch[1],
        'rpn_labels': next_batch[2],
        'rpn_deltas': next_batch[3],
        'rpn_mask': next_batch[4],
        'rpn_positive_range': next_batch[5],
        'rpn_positive_mask': next_batch[6],
        'cls': next_batch[7],
        'masks': next_batch[8],
        'valid_label_ranges': next_batch[9]
    }


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # img_in = tf.placeholder_v2(tf.float32, [None, None, None, 3])
    # anchor_type_in = tf.placeholder_v2(tf.int32, [None, None])
    # mask_in = tf.placeholder_v2(tf.int32, [None, config.RPN_ANCHORS_TRAIN_PER_IMAGE])
    # utils.generate_anchors()

    rpn_estimator = tf.estimator.Estimator(
        model_fn=rpn_model_fn,
        model_dir='models/rpn_model'
    )
    logging_hook = tf.train.LoggingTensorHook(
        tensors={
            "loss": "loss"
        }, every_n_iter=2)
    rpn_estimator.train(input_fn=train_input_fn, steps=40, hooks=[logging_hook])
    # inputs = train_input_fn()
    # output = rpn_model_fn(inputs[0], inputs[1], 0)
    # with tf.train.MonitoredSession() as sess:
    #     out = sess.run(output)
    #     proposals = out[0][0]
    #     print(proposals.shape)

    # dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.int32, tf.int32),
    #                                          (
    #                                              tf.TensorShape([None, None, 3]),
    #                                              tf.TensorShape([config.RPN_ANCHORS_TRAIN_PER_IMAGE, 2]),
    #                                              tf.TensorShape([None, 4]),
    #                                              tf.TensorShape([None]),
    #                                              tf.TensorShape([None])))
    # dataset = dataset.batch(2)
    # # dataset = dataset.prefetch(1)
    # next_batch = dataset.make_one_shot_iterator().get_next()
    # rpn_loss = RPN.rpn_loss(*next_batch)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     loss = sess.run(rpn_loss)
    #     print(loss[1])

