import os
import argparse
import numpy as np
import mmcv
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility
from pyquaternion import Quaternion
from shapely.geometry import Polygon


def box_corners_3d(center, size, yaw):
    l, w, h = size
    x, y, z = center
    corners = np.array([
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
    ])
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = (rot @ corners.T).T + np.array([x, y, z])
    return corners

def get_bottom_corners(box: Box):
    corners = box.corners()
    return corners[:, [0, 1, 2, 3]]

def get_bev_polygon(box: Box):
    corners = get_bottom_corners(box)[:2, :].T  # shape (4,2)
    return Polygon(corners)

def calculate_iou_bev(gt_box, pred_box):
    gt_polygon = get_bev_polygon(gt_box)
    pred_polygon = get_bev_polygon(pred_box)
    if not gt_polygon.is_valid or not pred_polygon.is_valid:
        return 0.0
    intersection = gt_polygon.intersection(pred_polygon).area
    union = gt_polygon.union(pred_polygon).area
    return intersection / union if union > 0 else 0.0

def calculate_iou_3d(gt_box, pred_box):
    bev_iou = calculate_iou_bev(gt_box, pred_box)
    z_min_gt = gt_box.center[2] - gt_box.wlh[2] / 2
    z_max_gt = gt_box.center[2] + gt_box.wlh[2] / 2
    z_min_pred = pred_box.center[2] - pred_box.wlh[2] / 2
    z_max_pred = pred_box.center[2] + pred_box.wlh[2] / 2

    z_overlap = max(0.0, min(z_max_gt, z_max_pred) - max(z_min_gt, z_min_pred))
    inter_volume = bev_iou * z_overlap
    vol_gt = np.prod(gt_box.wlh)
    vol_pred = np.prod(pred_box.wlh)
    union_vol = vol_gt + vol_pred - inter_volume
    return inter_volume / union_vol if union_vol > 0 else 0.0

def visualize_bev(sample_token, gt_boxes, pred_boxes, output_path):
    plt.figure(figsize=(10, 10))
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f"Bird's Eye View (Sample {sample_token})")

    for gt_box in gt_boxes:
        corners = get_bottom_corners(gt_box)
        plt.plot(corners[0, [0,1,2,3,0]], corners[1, [0,1,2,3,0]], color='green')

    for pred_box in pred_boxes:
        corners = get_bottom_corners(pred_box)
        plt.plot(corners[0, [0,1,2,3,0]], corners[1, [0,1,2,3,0]], color='red')

    green_line = plt.Line2D([0],[0],color='green', label='Ground Truth')
    red_line = plt.Line2D([0],[0],color='red', label='Prediction')
    plt.legend(handles=[green_line, red_line], loc='upper right')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def parse_predictions(predictions_file):
    outputs = mmcv.load(predictions_file)
    bbox_results = outputs.get('bbox_results', [])
    prediction_dict = {}
    for entry in bbox_results:
        token = entry.get('token')
        if token:
            boxes_3d = entry['boxes_3d'].tensor.cpu().numpy()  # Check shape here
            scores_3d = entry['scores_3d'].cpu().numpy()
            labels_3d = entry['labels_3d'].cpu().numpy()
            prediction_dict[token] = dict(
                boxes_3d=boxes_3d,
                scores_3d=scores_3d,
                labels_3d=labels_3d
            )
    return prediction_dict

def main(args):
    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=True)
    predictions = parse_predictions(args.predictions_file)

    total_iou_bev = []
    total_iou_3d = []

    os.makedirs(args.output_dir, exist_ok=True)

    scene_token_to_name = {sc['token']: sc['name'] for sc in nusc.scene}

    for idx, sample in enumerate(nusc.sample):
        sample_token = sample['token']

        # Ground truth boxes
        gt_boxes = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            center = ann['translation']
            # ann['size'] = [w, l, h]
            w, l, h = ann['size']
            orientation = Quaternion(ann['rotation'])
            gt_box = Box(
                center=center,
                size=np.array([w, l, h]),
                orientation=orientation,
                name=ann['category_name'],
                token=ann['token']
            )
            gt_boxes.append(gt_box)

        pred_boxes = []
        if sample_token in predictions:
            pred = predictions[sample_token]
            pred_boxes_3d = pred['boxes_3d']
            # Print shape to debug
            print(f"{sample_token}: pred_boxes_3d shape = {pred_boxes_3d.shape}")

            for i in range(pred_boxes_3d.shape[0]):
                # If there are more than 7 values, only take the first 7:
                # (x, y, z, dx, dy, dz, yaw) = pred_boxes_3d[i][:7]
                values = pred_boxes_3d[i]
                if len(values) < 7:
                    continue  # skip if not enough values
                x, y, z, dx, dy, dz, yaw = values[:7]

                # Reorder dx,dy,dz to w,l,h if necessary
                # Assuming dx=l, dy=w, dz=h:
                w, l, h = dy, dx, dz
                orientation = Quaternion(axis=[0,0,1], radians=yaw)
                pred_box = Box(
                    center=[x,y,z],
                    size=np.array([w, l, h]),
                    orientation=orientation,
                    name=f"Prediction {i}",
                )
                pred_boxes.append(pred_box)

        # Compute IoU
        for gt_box in gt_boxes:
            for pred_box in pred_boxes:
                iou_bev = calculate_iou_bev(gt_box, pred_box)
                iou_3d = calculate_iou_3d(gt_box, pred_box)
                total_iou_bev.append(iou_bev)
                total_iou_3d.append(iou_3d)

        output_path = os.path.join(args.output_dir, f'{sample_token}_bev.png')
        visualize_bev(sample_token, gt_boxes, pred_boxes, output_path)

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(nusc.sample)} samples...")

    avg_iou_bev = np.mean(total_iou_bev) if total_iou_bev else 0.0
    avg_iou_3d = np.mean(total_iou_3d) if total_iou_3d else 0.0

    print(f'Average IoU (BEV): {avg_iou_bev:.4f}')
    print(f'Average IoU (3D): {avg_iou_3d:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='Path to NuScenes dataset.')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to results.pkl.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save BEV images.')
    args = parser.parse_args()
    main(args)
