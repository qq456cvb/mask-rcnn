import tensorflow as tf
import tensorflow.contrib.slim as slim
import config


def forward_cls_bbox(rois):
    # reshape
    raw_size = tf.shape(rois)
    rois = tf.reshape(rois, [-1, config.CLS_BBOX_ROI_POOL_SIZE, config.CLS_BBOX_ROI_POOL_SIZE, 256])

    # class and box branch
    with tf.variable_scope('mrcnn_class_and_bbox'):
        x = slim.conv2d(rois, 256, [3, 3])
        x = slim.flatten(x)
        x = slim.fully_connected(x, 1024)
        x = slim.fully_connected(x, 1024)
        cls_logits = slim.fully_connected(x, config.NUM_CLASSES, activation_fn=None)
        cls = tf.nn.softmax(cls_logits)
        bbox_delta = slim.fully_connected(x, 4, activation_fn=None)
    return tf.reshape(cls_logits, [raw_size[0], -1, config.NUM_CLASSES]), tf.reshape(cls, [raw_size[0], -1, config.NUM_CLASSES]), tf.reshape(bbox_delta, [raw_size[0], -1, 4])


def forward_mask(rois):
    # reshape
    raw_size = tf.shape(rois)
    rois = tf.reshape(rois, [-1, config.MASK_ROI_POOL_SIZE, config.MASK_ROI_POOL_SIZE, 256])

    # mask branch
    with tf.variable_scope('mrcnn_mask'):
        x = slim.repeat(rois, 4, slim.conv2d, 256, [3, 3])
        x = slim.conv2d_transpose(x, 256, [2, 2], stride=2)
        mask_logits = slim.conv2d(x, config.NUM_CLASSES - 1, [1, 1], activation_fn=None)
        mask = tf.nn.sigmoid(mask_logits)
    return tf.reshape(mask_logits, [raw_size[0], -1, config.MASK_ROI_POOL_SIZE*2, config.MASK_ROI_POOL_SIZE*2, config.NUM_CLASSES - 1]),\
           tf.reshape(mask, [raw_size[0], -1, config.MASK_ROI_POOL_SIZE*2, config.MASK_ROI_POOL_SIZE*2, config.NUM_CLASSES - 1])


# we need rois and gt class id to get valid rois and positive rois respectively
# rois: [ROIS_PER_IMAGE * 4]
def mrcnn_loss(packed_values):
    pred_cls_logits, pred_delta, pred_mask_logits, gt_cls, gt_delta, gt_mask, rois = packed_values
    valid_roi_idx = tf.where(tf.not_equal(tf.reduce_sum(tf.abs(rois), axis=1), 0))
    pred_cls_logits = tf.gather_nd(pred_cls_logits, valid_roi_idx)
    pred_delta = tf.gather_nd(pred_delta, valid_roi_idx)
    pred_mask_logits = tf.gather_nd(pred_mask_logits, valid_roi_idx)
    gt_cls = tf.gather_nd(gt_cls, valid_roi_idx)
    gt_delta = tf.gather_nd(gt_delta, valid_roi_idx)
    gt_mask = tf.gather_nd(gt_mask, valid_roi_idx)

    # class loss is for all as negative rois should have class index 0
    gt_cls_onehot = tf.one_hot(gt_cls, config.NUM_CLASSES)
    gt_cls_onehot = tf.Print(gt_cls_onehot, [tf.shape(gt_cls_onehot)])
    cls_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_cls_onehot, logits=pred_cls_logits, name='mrcnn_class_loss')

    # delta loss is only for positive rois
    pos_roi_idx = tf.where(tf.not_equal(gt_cls, 0))
    # pos_roi_idx = tf.Print(pos_roi_idx, [tf.shape(pos_roi_idx)])
    pos_pred_delta = tf.gather_nd(pred_delta, pos_roi_idx)
    pos_gt_delta = tf.gather_nd(gt_delta, pos_roi_idx)
    bbox_abs_loss = tf.abs(pos_pred_delta - pos_gt_delta) - 0.5
    bbox_l2_loss = 0.5 * tf.square(bbox_abs_loss)
    bbox_loss = tf.where(tf.greater(bbox_abs_loss, 0.5), bbox_abs_loss, bbox_l2_loss, name='mrcnn_bbox_loss')

    # mask loss is also only for positive rois
    pos_pred_mask_logits = tf.gather_nd(pred_mask_logits, pos_roi_idx)
    pos_gt_mask = tf.gather_nd(gt_mask, pos_roi_idx)
    # filter predicted mask with ground truth class ids
    pos_gt_cls = tf.gather_nd(gt_cls, pos_roi_idx)
    pos_pred_mask_logits = tf.map_fn(lambda elem: tf.gather(elem[0], elem[1]-1, axis=-1), (pos_pred_mask_logits, pos_gt_cls), dtype=tf.float32)
    mask_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=pos_gt_mask, logits=pos_pred_mask_logits, name='mrcnn_mask_loss')

    return tf.reduce_sum(cls_loss), tf.reduce_sum(bbox_loss), tf.reduce_sum(mask_loss)



