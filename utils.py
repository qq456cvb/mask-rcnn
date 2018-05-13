import numpy as np
import config
import cv2
import tensorflow as tf
from pycocotools import mask as maskUtils


class BBOX_FORMAT:
    XYWH = 1
    YXYX = 2


def compute_delta(anchor, bbox):
    return np.concatenate([(bbox[:2]-anchor[:2]) / anchor[2:], np.log(bbox[2:] / anchor[2:])])


# apply k deltas to k anchors in tensors
# input shape: [K * 4]
# return [y1, x1, y2, x2] format
# anchors are in [y1, x1, y2, x2] format
# while deltas are in [dx, dy, dw, dh] format
def apply_delta_tf(anchors, deltas):
    w = anchors[:, 3] - anchors[:, 1]
    h = anchors[:, 2] - anchors[:, 0]
    y1 = h * deltas[:, 1] + anchors[:, 0]
    x1 = w * deltas[:, 0] + anchors[:, 1]
    y2 = y1 + h * tf.exp(deltas[:, 3])
    x2 = x1 + w * tf.exp(deltas[:, 2])
    return tf.stack([y1, x1, y2, x2], axis=1)


# pred_bbox is in YXYX format
# gt_bbox is in YXYX format
def calculate_delta_tf(pred_bbox, gt_bbox):
    w = pred_bbox[:, 3] - pred_bbox[:, 1]
    h = pred_bbox[:, 2] - pred_bbox[:, 0]
    dx = (gt_bbox[:, 1] - pred_bbox[:, 1]) / w
    dy = (gt_bbox[:, 0] - pred_bbox[:, 0]) / h
    dw = tf.log((gt_bbox[:, 3] - gt_bbox[:, 1]) / w)
    dh = tf.log((gt_bbox[:, 2] - gt_bbox[:, 0]) / h)
    return tf.stack([dx, dy, dw, dh], axis=1)


def log2_tf(x):
    return tf.log(x) / tf.log(tf.constant(2, dtype=x.dtype))


# tensorflow does not provide a trivial way to gather 2d elements according to the last dimension
def gather2d_tf(tensor, idx):
    idx_first = tf.transpose(tf.reshape(tf.tile(tf.range(tf.shape(idx)[0]), tf.shape(idx)[1:2]), [tf.shape(idx)[1], tf.shape(idx)[0]]))
    return tf.gather_nd(tensor, tf.concat([tf.expand_dims(idx_first, -1), tf.expand_dims(idx, -1)], -1))


def convert_bbox_tf(bbox, src=BBOX_FORMAT.YXYX):
    if src == BBOX_FORMAT.YXYX:
        w = bbox[:, 3] - bbox[:, 1]
        h = bbox[:, 2] - bbox[:, 0]
        return tf.stack([bbox[:, 1], bbox[:, 0], w, h], axis=1)
    elif src == BBOX_FORMAT.XYWH:
        y1 = bbox[:, 1]
        x1 = bbox[:, 0]
        y2 = y1 + bbox[:, 3]
        x2 = x1 + bbox[:, 2]
        return tf.stack([y1, x1, y2, x2], axis=1)
    else:
        raise Exception('convert bbox for src not implemented')


# this is all in unnormalized coordinates
def generate_anchors(format=BBOX_FORMAT.XYWH):
    anchors = []
    # every pyramid is assigned to one scale
    for scale_id, scale in enumerate(config.ANCHOR_SIZES):
        feature_map_size = config.IMG_SIZE_TRAIN // config.FEATURE_STRIDES[scale_id]
        anchors_lvl = np.zeros([feature_map_size, feature_map_size, config.ANCHORS_PER_PIXEL, 4])

        sqrt_area = np.sqrt(scale ** 2)
        width = sqrt_area / np.array(config.ANCHOR_RATIOS)
        height = sqrt_area * np.array(config.ANCHOR_RATIOS)

        xs, ys = np.meshgrid(np.arange(feature_map_size) * config.FEATURE_STRIDES[scale_id],
                             np.arange(feature_map_size) * config.FEATURE_STRIDES[scale_id])
        if format == BBOX_FORMAT.XYWH:
            anchors_lvl[:, :, :, 0] = np.expand_dims(xs, axis=2)
            anchors_lvl[:, :, :, 0] -= width / 2
            anchors_lvl[:, :, :, 1] = np.expand_dims(ys, axis=2)
            anchors_lvl[:, :, :, 1] -= height / 2

            anchors_lvl[:, :, :, 2] = width
            anchors_lvl[:, :, :, 3] = height
        elif format == BBOX_FORMAT.YXYX:
            anchors_lvl[:, :, :, 0] = np.expand_dims(ys, axis=2)
            anchors_lvl[:, :, :, 0] -= height / 2
            anchors_lvl[:, :, :, 1] = np.expand_dims(xs, axis=2)
            anchors_lvl[:, :, :, 1] -= width / 2

            anchors_lvl[:, :, :, 2] = anchors_lvl[:, :, :, 0] + height
            anchors_lvl[:, :, :, 3] = anchors_lvl[:, :, :, 1] + width
        anchors.append(anchors_lvl.reshape([-1, 4]))
    anchors = np.concatenate(anchors, axis=0)
    return anchors


def compute_ious(bbox, target_bboxs, target_bbox_areas=None):
    if target_bbox_areas is None:
        target_bbox_areas = target_bboxs[:, 2] * target_bboxs[:, 3]

    bbox_area = bbox[2] * bbox[3]
    x1 = np.maximum(bbox[0], target_bboxs[:, 0])
    x2 = np.minimum(bbox[0] + bbox[2], target_bboxs[:, 0] + target_bboxs[:, 2])
    y1 = np.maximum(bbox[1], target_bboxs[:, 1])
    y2 = np.minimum(bbox[1] + bbox[3], target_bboxs[:, 1] + target_bboxs[:, 3])
    intersect = (np.maximum(x2 - x1, 0)) * (np.maximum(y2 - y1, 0))
    union = bbox_area + target_bbox_areas - intersect
    return intersect / union


def compute_ious_tf(bbox, target_bboxs, target_bbox_areas):
    bbox_area = bbox[2] * bbox[3]
    x1 = tf.maximum(bbox[0], target_bboxs[:, 0])
    x2 = tf.minimum(bbox[0] + bbox[2], target_bboxs[:, 0] + target_bboxs[:, 2])
    y1 = tf.maximum(bbox[1], target_bboxs[:, 1])
    y2 = tf.minimum(bbox[1] + bbox[3], target_bboxs[:, 1] + target_bboxs[:, 3])
    intersect = (tf.maximum(x2 - x1, 0)) * (tf.maximum(y2 - y1, 0))
    union = bbox_area + target_bbox_areas - intersect
    return intersect / union


# compute overlaps between two sets of bboxs
def compute_overlaps(anchors, bboxs):
    result = np.zeros([anchors.shape[0], bboxs.shape[0]])
    anchor_areas = anchors[:, 2] * anchors[:, 3]
    for i in range(bboxs.shape[0]):
        result[:, i] = compute_ious(bboxs[i, :], anchors, anchor_areas)
    return result


def compute_overlaps_tf(anchors, bboxs):
    bbox_areas = bboxs[:, 2] * bboxs[:, 3]
    return tf.map_fn(lambda anchor: compute_ious_tf(anchor, bboxs, bbox_areas), anchors, dtype=tf.float32)


# reference https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
def resize_keep_ratio(img, target_size):
    target_size = np.array(target_size)
    old_size = img.shape[:2]
    ratio = np.amin(target_size / old_size)
    new_size = [int(x * ratio) for x in old_size]
    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta = target_size - np.array(new_size)
    top, bottom = delta[0] // 2, (delta[0] // 2) if (delta[0] & 1 == 0) else (delta[0] // 2 + 1)
    left, right = delta[1] // 2, (delta[1] // 2) if (delta[1] & 1 == 0) else (delta[1] // 2 + 1)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return ratio, img, np.array([left, top])


def generate_anchor_types(anchors, bboxs):
    if bboxs.size == 0:
        return -np.ones([anchors.shape[0]], dtype=np.int32), dict()
    overlaps = compute_overlaps(anchors, bboxs)

    anchor_types = np.zeros([overlaps.shape[0]], dtype=np.int32)

    anchor_idx = np.where(np.max(overlaps, axis=1) > 0.7)[0]
    anchor_types[anchor_idx] = 1
    matched_idx = np.argmax(overlaps, axis=1)
    match = dict(zip(anchor_idx, matched_idx[anchor_idx]))
    anchor_types[np.max(overlaps, axis=1) < 0.3] = -1
    # assign at least one box to gt box
    assigned_box_idx = np.argmax(overlaps, axis=0)
    anchor_types[assigned_box_idx] = 1
    for i in range(assigned_box_idx.size):
        match[assigned_box_idx[i]] = i

    return anchor_types, match


# return positive masks and all masks
def get_mask(anchor_types):
    positive_idx = np.where(anchor_types == 1)[0]
    if positive_idx.size > config.RPN_ANCHORS_TRAIN_PER_IMAGE // 2:
        positive_idx = np.random.choice(positive_idx, config.RPN_ANCHORS_TRAIN_PER_IMAGE // 2, replace=False)
    negative_idx = np.where(anchor_types == -1)[0]
    negative_idx = np.random.choice(negative_idx, config.RPN_ANCHORS_TRAIN_PER_IMAGE - positive_idx.size, replace=False)
    return positive_idx, np.concatenate((positive_idx, negative_idx))


def generate_rpn_labels(anchor_types, mask):
    sub_anchor_types = anchor_types[mask]
    sub_anchor_types[sub_anchor_types == -1] = 0
    labels = np.zeros([mask.shape[0], 2])
    labels[np.arange(labels.shape[0]), sub_anchor_types] = 1
    return labels


# assume mask idx sorted
def generate_rpn_deltas(anchors, bboxs, pos_mask, match):
    deltas = np.zeros([pos_mask.shape[0], 4])
    for i in range(pos_mask.shape[0]):
        deltas[i] = compute_delta(anchors[pos_mask[i]], bboxs[match[pos_mask[i]]])
    return deltas


# proposal from rpn, feed to mask rcnn
# proposal in YXYX format
# bbox in XYWH format
# bbox class should be fed by offset one since zero is for background
def generate_mask_rcnn_x_y_tf(packed_value):
    proposals, bbox, bbox_cls, bbox_mask, valid_label_range = packed_value

    # trim to valid_label_range
    valid_label_range = tf.Print(valid_label_range, [valid_label_range])
    bbox = bbox[:valid_label_range, :]
    bbox_cls = bbox_cls[:valid_label_range]
    bbox_mask = bbox_mask[:valid_label_range, :, :]

    # proposal shape [P * 4]
    # remove zero values
    mask = tf.cast(tf.reduce_sum(tf.abs(proposals), axis=1), tf.bool)
    proposals = tf.boolean_mask(proposals, mask)
    # proposals = tf.Print(proposals, [tf.shape(proposals)])

    # bbox shape [T * 4]
    # bbox class shape [T]
    # bbox mask shape [T * H * W]
    # overlaps shape [P * T]
    # bbox_cls_onehot = tf.one_hot(bbox_cls, depth=config.NUM_CLASSES)
    # proposals = convert_bbox_tf(tf.tile(tf.expand_dims(tf.constant([66.864, 665.936, 213.44,  47.28]), 0), [1000, 1]), src=BBOX_FORMAT.XYWH)
    overlaps = compute_overlaps_tf(convert_bbox_tf(proposals, src=BBOX_FORMAT.YXYX), bbox)
    bbox = convert_bbox_tf(bbox, src=BBOX_FORMAT.XYWH)
    max_overlaps = tf.reduce_max(overlaps, axis=1)

    # [Npos]
    positive_proposal_idx = tf.squeeze(tf.where(tf.greater(max_overlaps, 0.5)), axis=1)
    # positive_proposal_idx = tf.Print(positive_proposal_idx, [tf.shape(positive_proposal_idx)])
    # subsample according to 1:3 ratio
    target_positive_cnt = int(config.TRAIN_ROIS_PER_IMAGE - config.TRAIN_ROIS_PER_IMAGE / (1 + config.ROIS_POS2NEG_RATIO))
    positive_proposal_idx = tf.random_shuffle(positive_proposal_idx)[:target_positive_cnt]

    # [Nneg]
    negative_proposal_idx = tf.squeeze(tf.where(tf.less_equal(max_overlaps, 0.5)), axis=1)
    negative_proposal_idx = tf.random_shuffle(negative_proposal_idx)[:config.TRAIN_ROIS_PER_IMAGE - tf.shape(positive_proposal_idx)[0]]
    # negative_proposal_idx = tf.Print(negative_proposal_idx, [tf.shape(negative_proposal_idx)])
    overlaps_bbox_idx = tf.argmax(overlaps, axis=1)

    # pick positive assignment idx
    positive_overlaps_bbox_idx = tf.gather(overlaps_bbox_idx, positive_proposal_idx)
    negative_overlaps_bbox_idx = tf.gather(overlaps_bbox_idx, negative_proposal_idx)
    target_bbox_cls = tf.gather(bbox_cls, positive_overlaps_bbox_idx, axis=0)
    # target_bbox_cls = tf.Print(target_bbox_cls, [tf.shape(target_bbox_cls)])
    # fill negative class with label 0
    target_bbox_cls = tf.pad(target_bbox_cls,
                                [[0, config.TRAIN_ROIS_PER_IMAGE - tf.shape(target_bbox_cls)[0]]])
    # target_bbox_labels = tf.one_hot(target_bbox_labels, depth=config.NUM_CLASSES)

    positive_proposals = tf.gather(proposals, positive_proposal_idx, axis=0)
    positive_bbox = tf.gather(bbox, positive_overlaps_bbox_idx, axis=0)
    negative_proposals = tf.gather(proposals, negative_proposal_idx, axis=0)
    negative_bbox = tf.gather(bbox, negative_overlaps_bbox_idx, axis=0)
    target_bbox_deltas = calculate_delta_tf(positive_proposals, positive_bbox)
    target_bbox_deltas = tf.pad(target_bbox_deltas,
                                [[0, config.TRAIN_ROIS_PER_IMAGE - tf.shape(target_bbox_deltas)[0]], [0, 0]])
    if config.USE_GT_BBOX_IN_MASK_RCNN:
        target_bbox_deltas = tf.zeros([config.TRAIN_ROIS_PER_IMAGE , 4])

    positive_rois = positive_bbox if config.USE_GT_BBOX_IN_MASK_RCNN else positive_proposals
    negative_rois = negative_bbox if config.USE_GT_BBOX_IN_MASK_RCNN else negative_proposals

    rois = tf.concat([positive_rois, negative_rois], axis=0)
    # if there are not enough proposals, just pad rois
    rois = tf.pad(rois, [[0, config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0]], [0, 0]])
    # rois = tf.Print(rois, [tf.shape(rois)])

    target_bbox_mask = tf.gather(bbox_mask, positive_overlaps_bbox_idx, axis=0)
    y1 = (positive_proposals[:, 0] - positive_bbox[:, 0]) / config.MASK_OUTPUT_SHAPE
    x1 = (positive_proposals[:, 1] - positive_bbox[:, 1]) / config.MASK_OUTPUT_SHAPE
    y2 = (positive_proposals[:, 2] - positive_bbox[:, 0]) / config.MASK_OUTPUT_SHAPE
    x2 = (positive_proposals[:, 3] - positive_bbox[:, 1]) / config.MASK_OUTPUT_SHAPE
    target_bbox_mask = tf.squeeze(tf.image.crop_and_resize(tf.expand_dims(target_bbox_mask, -1), tf.stack([y1, x1, y2, x2], axis=1), tf.range(tf.shape(target_bbox_mask)[0]),
                                                [config.MASK_OUTPUT_SHAPE, config.MASK_OUTPUT_SHAPE]), -1)

    # fill mask with all zeros
    target_bbox_mask = tf.pad(target_bbox_mask,
                              [[0, config.TRAIN_ROIS_PER_IMAGE - tf.shape(target_bbox_mask)[0]], [0, 0], [0, 0]])

    return rois, target_bbox_cls, target_bbox_deltas, target_bbox_mask


# adapted from https://github.com/matterport/Mask_RCNN/blob/master/coco.py
def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m


if __name__ == '__main__':
    a = tf.constant([[2, 0], [0, 1]])
    b = tf.gather_nd(a, tf.where(a > 0))
    with tf.Session() as sess:
        print(sess.run(b))
    # a = np.arange(6).reshape([3, 2])
    # print(a[[0, 2]])
    # bbox = np.array([0, 0, 512, 512])
    # target_bboxs = np.array([
    #     [3, 0, 590, 570]
    # ])
    # print(compute_ious(bbox, target_bboxs))