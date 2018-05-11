import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import ResNet_w_FPN
import config
import utils


def forward(feature_maps):
    batch_size = tf.shape(feature_maps[0])[0]
    all_rpn_anchor_logits = []
    all_rpn_anchor_probs = []
    all_rpn_bbox_delta = []
    for i, feature_level in enumerate(feature_maps):
        features = slim.conv2d(inputs=feature_level, num_outputs=512, kernel_size=3, reuse=(None if i == 0 else True), scope='rpn_feature_extract')
        # features = tf.Print(features, [tf.shape(features)])
        rpn_anchor_logits = slim.conv2d(inputs=features, num_outputs=2 * config.ANCHORS_PER_PIXEL, kernel_size=1, activation_fn=None)

        # reshape for softmax
        rpn_anchor_logits = tf.reshape(rpn_anchor_logits,
                                           [tf.shape(rpn_anchor_logits)[0], tf.shape(rpn_anchor_logits)[1], tf.shape(rpn_anchor_logits)[2], config.ANCHORS_PER_PIXEL, 2])
        all_rpn_anchor_logits.append(tf.reshape(rpn_anchor_logits, [batch_size, -1, 2]))
        # [H * W * A * 2]
        rpn_anchor_probs = tf.nn.softmax(rpn_anchor_logits)
        all_rpn_anchor_probs.append(tf.reshape(rpn_anchor_probs, [batch_size, -1, 2]))

        rpn_bbox_delta = slim.conv2d(inputs=features, num_outputs=4 * config.ANCHORS_PER_PIXEL, kernel_size=1, activation_fn=None)
        all_rpn_bbox_delta.append(tf.reshape(rpn_bbox_delta, [batch_size, -1, 4]))

    return tf.concat(all_rpn_anchor_logits, axis=1), tf.concat(all_rpn_anchor_probs, axis=1), tf.concat(all_rpn_bbox_delta, axis=1)


def inference(packed_values):
    pred_rpn_anchor_probs, pred_rpn_anchor_deltas, all_anchors = packed_values
    # all anchors should be served as a static tensor and in [y1, x1, y2, x2] format
    best_k_anchor_scores, best_k_anchor_idx = tf.nn.top_k(pred_rpn_anchor_probs[:, 1], config.RPN_TOP_K_FOR_TRAINING)
    # best k anchor idx: [K]
    best_k_anchor_deltas = tf.gather(pred_rpn_anchor_deltas, best_k_anchor_idx)
    # get bbox in [y1, x1, y2, x2] format
    # [K * 4]
    reference_k_bbox = tf.gather(all_anchors, best_k_anchor_idx)
    best_k_anchor_bbox = utils.apply_delta_tf(reference_k_bbox, best_k_anchor_deltas)
    # best_k_anchor_bbox = reference_k_bbox
    # [1000]
    post_nms_idx_in_best_k = tf.image.non_max_suppression(best_k_anchor_bbox, best_k_anchor_scores, config.RPN_PROPOSEL_NUM, iou_threshold=0.5)
    post_nms_idx_in_best_k = tf.pad(post_nms_idx_in_best_k, [[0, config.RPN_PROPOSEL_NUM - tf.shape(post_nms_idx_in_best_k)[0]]], constant_values=-1)
    # since on GPU, negative index will fill tensor with zero
    post_nms_bbox = tf.gather(best_k_anchor_bbox, post_nms_idx_in_best_k, axis=0)
    # return bbox in un-normalized YXYX format
    post_nms_bbox = tf.maximum(tf.minimum(post_nms_bbox, config.IMG_SIZE_TRAIN - 1), 0)
    return post_nms_bbox


def rpn_loss(packed_values):
    pred_rpn_anchor_logits, pred_rpn_anchor_deltas, target_rpn_anchor_labels, target_rpn_anchor_deltas, mask, positive_range, positive_mask = packed_values
    class_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_rpn_anchor_labels,
                                                            logits=tf.gather(pred_rpn_anchor_logits, mask), name='rpn_class_loss')
    bbox_abs_loss = tf.abs(target_rpn_anchor_deltas[:positive_range, :] - tf.gather(pred_rpn_anchor_deltas, tf.gather_nd(positive_mask, tf.where(tf.not_equal(positive_mask, -1))))) - 0.5
    bbox_l2_loss = 0.5 * tf.square(bbox_abs_loss)
    bbox_loss = tf.where(tf.greater(bbox_abs_loss, 0.5), bbox_abs_loss, bbox_l2_loss, name='rpn_bbox_loss')
    return tf.reduce_sum(class_loss), tf.reduce_sum(bbox_loss)


if __name__ == '__main__':
    print(config.ANCHORS_PER_PIXEL)