import tensorflow as tf
import utils
import config


# generate roi with roi align layer
# feature maps P2 P3 P4 P5 list of [B * H * W * C]
def forward(proposals, feature_maps, pool_size):
    # proposal shape: [B * P * 4], YXYX format
    w = proposals[:, :, 3] - proposals[:, :, 1]
    h = proposals[:, :, 2] - proposals[:, :, 0]

    # making roi resolutions roughly the same, while they differ in receptive fields
    lvls = tf.floor(2 + utils.log2_tf(w * h / tf.constant(224., dtype=tf.float32)))
    lvls = tf.cast(tf.maximum(tf.minimum(lvls, 3), 0), tf.int32)
    roi_pooled_all = []
    level_idx_all = []
    for i in range(4):
        feature_map = feature_maps[i]

        # [N * 2]
        level_idx = tf.where(tf.equal(lvls, i))
        level_bbox = tf.gather_nd(proposals, level_idx)
        level_bbox = level_bbox / config.IMG_SIZE_TRAIN

        # crop and resize
        batch_ind = tf.cast(level_idx[:, 0], tf.int32)
        if not config.BACKPROP_ROI:
            level_bbox = tf.stop_gradient(level_bbox)
            batch_ind = tf.stop_gradient(batch_ind)
        roi_pooled = tf.image.crop_and_resize(feature_map, level_bbox, batch_ind,
                                              [pool_size * config.ROI_MAX_IN_POOL_SIZE,
                                               pool_size * config.ROI_MAX_IN_POOL_SIZE])
        roi_pooled_all.append(roi_pooled)
        level_idx_all.append(level_idx)
    roi_pooled_all = tf.concat(roi_pooled_all, axis=0)
    level_idx_all = tf.concat(level_idx_all, axis=0)
    level_idx_1d = level_idx_all[:, 0] * (tf.reduce_max(level_idx_all[:, 1]) + 1) + level_idx_all[:, 1]
    # we want ascend order so negate the input
    _, sorted_idx = tf.nn.top_k(-level_idx_1d, k=tf.shape(level_idx_1d)[0])
    roi_pooled_final = tf.gather(roi_pooled_all, sorted_idx)
    if config.ROI_MAX_IN_POOL_SIZE > 1:
        roi_pooled_final = tf.nn.max_pool(roi_pooled_final, ksize=[1, config.ROI_MAX_IN_POOL_SIZE, config.ROI_MAX_IN_POOL_SIZE, 1],
                                          strides=[1, config.ROI_MAX_IN_POOL_SIZE, config.ROI_MAX_IN_POOL_SIZE, 1], padding='SAME')
    batch_size = tf.shape(proposals)[0]
    shape = tf.shape(roi_pooled_final)
    return tf.reshape(roi_pooled_final, [batch_size, -1, shape[1], shape[2], shape[3]])

