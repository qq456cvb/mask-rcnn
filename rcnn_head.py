import tensorflow as tf
import tensorflow.contrib.slim as slim
import config


def forward_cls_bbox(rois):
    # reshape
    raw_size = tf.shape(rois)
    rois = tf.reshape(rois, [-1, raw_size[2], raw_size[3], 256])

    # class and box branch
    with tf.variable_scope('mrcnn_class_and_bbox'):
        x = slim.conv2d(rois, 256, [3, 3])
        x = slim.flatten(x)
        x = slim.fully_connected(x, 1024)
        x = slim.fully_connected(x, 1024)
        cls = slim.fully_connected(x, config.NUM_CLASSES, activation_fn=tf.nn.softmax)
        bbox = slim.fully_connected(x, 4, activation_fn=None)
    return tf.reshape(cls, [raw_size[0], -1, config.NUM_CLASSES]), tf.reshape(bbox, [raw_size[0], -1, 4])


def forward_mask(rois):
    # reshape
    raw_size = tf.shape(rois)
    rois = tf.reshape(rois, [-1, raw_size[2], raw_size[3], 256])

    # mask branch
    with tf.variable_scope('mrcnn_mask'):
        x = slim.repeat(rois, 4, slim.conv2d, 256, [3, 3])
        x = slim.conv2d_transpose(x, 256, [2, 2], stride=2)
        mask = slim.conv2d(x, config.NUM_CLASSES - 1, [1, 1], activation_fn=tf.nn.sigmoid)
    return tf.reshape(mask, [raw_size[0], -1, config.MASK_ROI_POOL_SIZE*2, config.MASK_ROI_POOL_SIZE*2, config.NUM_CLASSES - 1])