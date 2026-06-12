# mask-rcnn

A **tiny Mask R-CNN** ([He et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)) implemented from scratch in TensorFlow 1.x — the whole model in roughly 700 lines, spread over a handful of small, readable modules. The trick that keeps it compact is `tf.map_fn`: all per-image logic (RPN losses, proposal decoding, ROI target assignment, head losses) is written for a single image and mapped over the batch, instead of the heavily vectorized batch code found in larger implementations.

The goal is readability — a minimal reference for how the pieces of Mask R-CNN fit together, trained on COCO.

## Code Map

| File | Role |
| --- | --- |
| `ResNet_w_FPN.py` | ResNet-style backbone with a Feature Pyramid Network producing multi-scale feature maps (strides 4–64). |
| `RPN.py` | Region Proposal Network: objectness + anchor deltas per FPN level, loss, and proposal decoding with NMS. |
| `RoiAlign.py` | ROI Align via `tf.image.crop_and_resize`, with ROIs routed to the right FPN level by their scale. |
| `rcnn_head.py` | The two heads: classification + box refinement (FC) and the mask branch (4 convs + deconv, 28×28 per-class masks). |
| `utils.py` | Anchor generation, anchor/GT matching, RPN and ROI target computation, COCO mask decoding. |
| `generator.py` | COCO data generator: loads images and annotations with `pycocotools`, resizes to 1024×1024 keeping aspect ratio, and precomputes RPN targets on the CPU. |
| `main.py` | Wires everything into a `tf.estimator.Estimator` training loop. |
| `config.py` | All hyperparameters (anchor sizes/ratios, ROI counts, pool sizes, dataset path). |

## Run

Requires TensorFlow 1.x, `pycocotools`, OpenCV, SciPy and matplotlib, plus a copy of [COCO](https://cocodataset.org/) (images + instance annotations).

1. Point `DATASET_DIR` / `DATASET_TYPE` in `config.py` at your COCO root and split (e.g. `val2017`).
2. Set `DEBUG = False` in `config.py` (when `True`, the generator pops up a matplotlib window visualizing each image's GT boxes and matched anchors — handy for sanity-checking targets, not for training).
3. Train:

```bash
python main.py
```

Checkpoints and TensorBoard logs go to `models/rpn_model`. Note `main.py` runs a short demonstration schedule (`steps=40`); raise the `steps` argument for a real training run.

## Status

This is an educational/prototype implementation: the training pipeline (RPN + heads, all five losses) is complete, but there is no standalone inference/evaluation script and no pretrained backbone loading — expect to tinker.

## Citation

```bibtex
@inproceedings{he2017mask,
  title={Mask R-CNN},
  author={He, Kaiming and Gkioxari, Georgia and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}
```
