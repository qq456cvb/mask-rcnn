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
        ratio, img = utils.resize_keep_ratio(img, (1024, 1024))

        ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=0)
        anns = coco.loadAnns(ann_ids)
        bboxs = [ann['bbox'] for ann in anns]
        bboxs = np.vstack(bboxs) * ratio

        # bboxs = bboxs[0:1, :]
        print(bboxs.shape)
        anchor_types, matches = utils.generate_anchor_types(all_anchors, bboxs)
        positive_mask, mask = utils.get_mask(anchor_types)
        labels = utils.generate_rpn_labels(anchor_types, mask)
        deltas = utils.generate_rpn_deltas(all_anchors, bboxs, positive_mask, matches)

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
        yield img, bboxs, labels, deltas, mask, positive_mask
