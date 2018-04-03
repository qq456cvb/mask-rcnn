from pycocotools.coco import COCO
import numpy as np
import config
import scipy.ndimage
import cv2
import utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def data_generator():
    ann_file = '{}/annotations/instances_{}.json'.format(config.DATASET_DIR, config.DATASET_TYPE)
    coco = COCO(ann_file)
    categories = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in categories]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    img_ids = coco.getImgIds()
    all_anchors = utils.generate_anchors()
    while True:
        rand = np.random.randint(0, len(img_ids))
        rand = 3118
        # print(rand)

        img_info = coco.loadImgs(img_ids[rand])[0]
        img = scipy.ndimage.imread(config.DATASET_DIR + '\\' + config.DATASET_TYPE + '\\' + img_info['file_name'])
        img = img.astype(np.float32) / 255.
        ratio, img, offset = utils.resize_keep_ratio(img, (1024, 1024))

        ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=0)
        anns = coco.loadAnns(ann_ids)
        bboxs = [ann['bbox'] for ann in anns]
        bboxs = np.vstack(bboxs)
        # OFFSET one for backgroound
        cls = np.array([ann['category_id'] + 1 for ann in anns])
        masks = np.array([utils.annToMask(ann, img_info['height'], img_info['width']) for ann in anns])

        # resize masks to desired shape
        bboxs_ind = bboxs.astype(np.int)
        masks = np.array([cv2.resize(mask[bboxs_ind[i, 1]:bboxs_ind[i, 1] + bboxs_ind[i, 3], bboxs_ind[i, 0]:bboxs_ind[i, 0] + bboxs_ind[i, 2]], (config.MASK_OUTPUT_SHAPE, config.MASK_OUTPUT_SHAPE))
                          for i, mask in enumerate(masks)])
        bboxs = bboxs * ratio
        bboxs[:, :2] += offset
        bboxs_rpn = bboxs

        valid_label_range = 0
        # we pad ot trim all labels to MAX_GT_TRAIN_INSTANCES to make it batched
        if bboxs.shape[0] > config.MAX_GT_TRAIN_INSTANCES:
            valid_label_range = config.MAX_GT_TRAIN_INSTANCES
            bboxs = bboxs[:config.MAX_GT_TRAIN_INSTANCES, :]
            cls = cls[:config.MAX_GT_TRAIN_INSTANCES]
            masks = masks[:config.MAX_GT_TRAIN_INSTANCES, :, :]
        else:
            valid_label_range = bboxs.shape[0]
            bboxs = np.pad(bboxs, ((0, config.MAX_GT_TRAIN_INSTANCES - bboxs.shape[0]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0)))
            cls = np.pad(cls, (0, config.MAX_GT_TRAIN_INSTANCES - cls.shape[0]), mode='constant', constant_values=(0, 0))
            masks = np.pad(masks, ((0, config.MAX_GT_TRAIN_INSTANCES - masks.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))

        # pre compute rpn targets
        anchor_types, matches = utils.generate_anchor_types(all_anchors, bboxs_rpn)
        rpn_positive_mask, rpn_mask = utils.get_mask(anchor_types)
        rpn_labels = utils.generate_rpn_labels(anchor_types, rpn_mask)
        rpn_deltas = utils.generate_rpn_deltas(all_anchors, bboxs_rpn, rpn_positive_mask, matches)

        if config.DEBUG:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.imshow(img)
            # coco.showAnns(anns)
            for bbox in bboxs:
                ax.add_patch(
                    patches.Rectangle(
                        (bbox[0], bbox[1]),
                        bbox[2],
                        bbox[3],
                        edgecolor="red",
                        fill=False  # remove background
                    )
                )
            for m in matches:
                ax.add_patch(
                            patches.Rectangle(
                                (all_anchors[m][0], all_anchors[m][1]),
                                all_anchors[m][2],
                                all_anchors[m][3],
                                edgecolor="blue",
                                fill=False  # remove background
                            )
                )
            plt.show()
        # we feed precomputed rpn masks on multi-threaded cpu
        yield img, bboxs, rpn_labels, rpn_deltas, rpn_mask, rpn_positive_mask, cls, masks, valid_label_range
