import os
import sys
import mmcv
import torch
import logging
import importlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from pyquaternion import Quaternion
from shapely.geometry import Polygon

from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility, view_points
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from mmdet3d.datasets.nuscenes_dataset import output_to_nusc_box, lidar_nusc_box_to_global


# ----------------------------
# Configuration Parameters
# ----------------------------

CONFIG_FILE = '/path/to/your/config/file.py'
CHECKPOINT_FILE = '/path/to/your/checkpoint.pth'
DATA_ROOT = '/path/to/your/nuscenes/data'
PREDICTIONS_FILE = '/path/to/your/results.pkl'
SHOW_DIR = '/path/to/your/output/visualizations'

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

IOU_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD = 0.7

# ----------------------------
# Logging Configuration
# ----------------------------
os.makedirs(SHOW_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(os.path.join(SHOW_DIR, 'inference.log'), mode='w')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

metrics_logger = logging.getLogger('metrics_logger')
metrics_logger.setLevel(logging.INFO)
metrics_file_handler = logging.FileHandler(os.path.join(SHOW_DIR, 'metrics.log'), mode='w')
metrics_logger.addHandler(metrics_file_handler)

# ----------------------------
# Utility Functions
# ----------------------------

def calculate_iou_3d(box1, box2):
    poly1 = Polygon(box1.bottom_corners()[:2, :].T)
    poly2 = Polygon(box2.bottom_corners()[:2, :].T)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    z_min1, z_max1 = box1.center[2] - box1.wlh[2] / 2.0, box1.center[2] + box1.wlh[2] / 2.0
    z_min2, z_max2 = box2.center[2] - box2.wlh[2] / 2.0, box2.center[2] + box2.wlh[2] / 2.0
    z_overlap = max(0.0, min(z_max1, z_max2) - max(z_min1, z_min2))
    if z_overlap == 0.0:
        return 0.0
    intersection_volume = intersection * z_overlap
    volume1 = np.prod(box1.wlh)
    volume2 = np.prod(box2.wlh)
    union_volume = volume1 + volume2 - intersection_volume
    return intersection_volume / union_volume if union_volume > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes, iou_threshold=IOU_THRESHOLD):
    matched_gt, matched_pred = [], []
    unmatched_gt, unmatched_pred = [], []

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou_3d(gt, pred)

    cost_matrix = -iou_matrix
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    for g_idx, p_idx in zip(gt_indices, pred_indices):
        if iou_matrix[g_idx, p_idx] >= iou_threshold:
            matched_gt.append(gt_boxes[g_idx])
            matched_pred.append(pred_boxes[p_idx])
        else:
            unmatched_gt.append(gt_boxes[g_idx])
            unmatched_pred.append(pred_boxes[p_idx])

    unmatched_gt += [gt_boxes[i] for i in range(len(gt_boxes)) if i not in gt_indices]
    unmatched_pred += [pred_boxes[i] for i in range(len(pred_boxes)) if i not in pred_indices]

    return matched_gt, matched_pred, unmatched_gt, unmatched_pred


def load_unidad_predictions(predictions_file):
    outputs = mmcv.load(predictions_file)
    bbox_results = outputs['bbox_results']
    predictions = []
    for result in bbox_results:
        bboxes = result['boxes_3d'].tensor.numpy()
        scores = result['scores_3d'].numpy()
        labels = result['labels_3d'].numpy()
        boxes = []
        for i, bbox in enumerate(bboxes):
            if scores[i] < CONFIDENCE_THRESHOLD:
                continue
            box = Box(center=bbox[:3], size=bbox[3:6], orientation=Quaternion(axis=[0, 0, 1], radians=bbox[6]))
            box.label = labels[i]
            box.score = scores[i]
            box.name = CLASS_NAMES[labels[i]] if labels[i] < len(CLASS_NAMES) else 'unknown'
            boxes.append(box)
        predictions.append(boxes)
    return predictions


# ----------------------------
# Main Function
# ----------------------------

def main():
    cfg = Config.fromfile(CONFIG_FILE)
    dataset = build_dataset(cfg.data.test)
    data_infos = dataset.data_infos

    predictions = load_unidad_predictions(PREDICTIONS_FILE)
    total_iou_3d, total_iou_bev = [], []
    true_positives, false_positives, false_negatives = 0, 0, 0

    for idx, data_info in enumerate(data_infos):
        gt_boxes = [Box(center=ann['translation'], size=ann['size'], orientation=Quaternion(ann['rotation'])) for ann in dataset.get_ann_info(idx)['gt_bboxes_3d']]
        pred_boxes = predictions[idx]

        matched_gt, matched_pred, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes)

        true_positives += len(matched_gt)
        false_positives += len(unmatched_pred)
        false_negatives += len(unmatched_gt)

        for gt, pred in zip(matched_gt, matched_pred):
            iou = calculate_iou_3d(gt, pred)
            total_iou_3d.append(iou)

    if total_iou_3d:
        avg_iou_3d = sum(total_iou_3d) / len(total_iou_3d)
        logger.info(f"Average IoU 3D: {avg_iou_3d:.2f}")
        metrics_logger.info(f"Average IoU 3D: {avg_iou_3d:.2f}")

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
    logger.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
    metrics_logger.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}")


if __name__ == '__main__':
    main()
