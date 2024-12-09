import cv2
import torch
import argparse
import os
import glob
import numpy as np
import mmcv
import matplotlib
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.utils.data_classes import Box
from nuscenes.utils import splits
from pyquaternion import Quaternion
from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap
from PIL import Image
from tools.analysis_tools.visualize.utils import AgentPredictionData
from tools.analysis_tools.visualize.render.bev_render import BEVRender
from tools.analysis_tools.visualize.render.cam_render import CameraRender
from shapely.geometry import Polygon


def box_corners_3d(center, dims, yaw):
    # Convert box center, dims, yaw to 3D corners
    l, w, h = dims
    x, y, z = center
    corners = np.array([
        [l/2,  w/2,  h/2],
        [l/2,  w/2, -h/2],
        [l/2, -w/2,  h/2],
        [l/2, -w/2, -h/2],
        [-l/2, w/2,  h/2],
        [-l/2, w/2, -h/2],
        [-l/2,-w/2,  h/2],
        [-l/2,-w/2, -h/2]
    ])
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = (rot @ corners.T).T + np.array([x, y, z])
    return corners

def bev_box_2d(corners_3d):
    # Extract the BEV bounding box from 3D corners
    corners_xy = corners_3d[:, [0,1]]
    xmin, xmax = np.min(corners_xy[:,0]), np.max(corners_xy[:,0])
    ymin, ymax = np.min(corners_xy[:,1]), np.max(corners_xy[:,1])
    box2d = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]
    ])
    return box2d

def box_iou_bev(box1_corners, box2_corners):
    poly1 = Polygon(box1_corners)
    poly2 = Polygon(box2_corners)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def box_iou_3d(gt_box, pred_box):
    # Compute 3D IoU by extruding BEV intersection
    gt_corners = box_corners_3d(gt_box['center'], gt_box['dims'], gt_box['yaw'])
    pd_corners = box_corners_3d(pred_box['center'], pred_box['dims'], pred_box['yaw'])
    gt_bev = bev_box_2d(gt_corners)
    pd_bev = bev_box_2d(pd_corners)
    bev_iou = box_iou_bev(gt_bev, pd_bev)

    gt_zmin = np.min(gt_corners[:,2])
    gt_zmax = np.max(gt_corners[:,2])
    pd_zmin = np.min(pd_corners[:,2])
    pd_zmax = np.max(pd_corners[:,2])
    inter_h = max(0, min(gt_zmax, pd_zmax) - max(gt_zmin, pd_zmin))
    if inter_h <= 0:
        return 0.0, bev_iou

    poly_gt = Polygon(gt_bev)
    poly_pd = Polygon(pd_bev)
    inter_area = poly_gt.intersection(poly_pd).area
    inter_vol = inter_area * inter_h
    gt_vol = np.prod(gt_box['dims'])
    pd_vol = np.prod(pred_box['dims'])
    union_vol = gt_vol + pd_vol - inter_vol
    iou_3d = inter_vol / union_vol if union_vol > 0 else 0.0
    return iou_3d, bev_iou


class Visualizer:
    """
    Visualizer class that plots ground truth boxes in green, predicted boxes in red,
    and computes average 3D and BEV IoU.
    """

    def __init__(
            self,
            dataroot='data/nuscenes',
            version='v1.0-mini',
            predroot=None,
            with_occ_map=False,
            with_map=False,
            with_planning=False,
            with_pred_box=True,
            with_pred_traj=False,
            show_gt_boxes=True,
            show_lidar=False,
            show_command=False,
            show_hd_map=False,
            show_sdc_car=False,
            show_sdc_traj=False,
            show_legend=False):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.predict_helper = PredictHelper(self.nusc)
        self.with_occ_map = with_occ_map
        self.with_map = with_map
        self.with_planning = with_planning
        self.show_lidar = show_lidar
        self.show_command = show_command
        self.show_hd_map = show_hd_map
        self.show_sdc_car = show_sdc_car
        self.show_sdc_traj = show_sdc_traj
        self.show_legend = show_legend
        self.with_pred_traj = with_pred_traj
        self.with_pred_box = with_pred_box
        self.show_gt_boxes = show_gt_boxes
        self.veh_id_list = [0, 1, 2, 3, 4, 6, 7]
        self.use_json = '.json' in predroot if predroot is not None else False
        self.token_set = set()
        self.predictions = self._parse_predictions_multitask_pkl(predroot)
        self.bev_render = BEVRender(show_gt_boxes=show_gt_boxes)
        self.cam_render = CameraRender(show_gt_boxes=show_gt_boxes)

        if self.show_hd_map:
            self.nusc_maps = {
                'boston-seaport': NuScenesMap(dataroot=dataroot, map_name='boston-seaport'),
                'singapore-hollandvillage': NuScenesMap(dataroot=dataroot, map_name='singapore-hollandvillage'),
                'singapore-onenorth': NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth'),
                'singapore-queenstown': NuScenesMap(dataroot=dataroot, map_name='singapore-queenstown'),
            }

        self.all_3d_ious = []
        self.all_bev_ious = []

    def _parse_predictions_multitask_pkl(self, predroot):
        if predroot is None:
            return dict()
        outputs = mmcv.load(predroot)
        outputs = outputs['bbox_results']
        prediction_dict = dict()
        for k in range(len(outputs)):
            token = outputs[k]['token']
            self.token_set.add(token)
            if self.show_sdc_traj:
                outputs[k]['boxes_3d'].tensor = torch.cat(
                    [outputs[k]['boxes_3d'].tensor, outputs[k]['sdc_boxes_3d'].tensor], dim=0)
                outputs[k]['scores_3d'] = torch.cat(
                    [outputs[k]['scores_3d'], outputs[k]['sdc_scores_3d']], dim=0)
                outputs[k]['labels_3d'] = torch.cat([outputs[k]['labels_3d'], torch.zeros(
                    (1,), device=outputs[k]['labels_3d'].device)], dim=0)
            bboxes = outputs[k]['boxes_3d']
            scores = outputs[k]['scores_3d']
            labels = outputs[k]['labels_3d']

            track_scores = scores.cpu().detach().numpy()
            track_labels = labels.cpu().detach().numpy()
            track_centers = bboxes.gravity_center.cpu().detach().numpy()
            track_dims = bboxes.dims.cpu().detach().numpy()
            track_yaw = bboxes.yaw.cpu().detach().numpy()

            if 'track_ids' in outputs[k]:
                track_ids = outputs[k]['track_ids'].cpu().detach().numpy()
            else:
                track_ids = None

            track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]
            trajs = outputs[k]['traj'].numpy()
            traj_scores = outputs[k]['traj_scores'].numpy()

            predicted_agent_list = []

            if self.with_occ_map:
                if 'topk_query_ins_segs' in outputs[k]['occ']:
                    occ_map = outputs[k]['occ']['topk_query_ins_segs'][0].cpu().numpy()
                else:
                    occ_map = np.zeros((1, 5, 200, 200))
            else:
                occ_map = None

            occ_idx = 0
            for i in range(track_scores.shape[0]):
                if track_scores[i] < 0.25:
                    continue
                if occ_map is not None and track_labels[i] in self.veh_id_list:
                    occ_map_cur = occ_map[occ_idx, :, ::-1]
                    occ_idx += 1
                else:
                    occ_map_cur = None
                if track_ids is not None:
                    if i < len(track_ids):
                        track_id = track_ids[i]
                    else:
                        track_id = 0
                else:
                    track_id = None
                predicted_agent_list.append(
                    AgentPredictionData(
                        track_scores[i],
                        track_labels[i],
                        track_centers[i],
                        track_dims[i],
                        track_yaw[i],
                        track_velocity[i],
                        trajs[i],
                        traj_scores[i],
                        pred_track_id=track_id,
                        pred_occ_map=occ_map_cur,
                        past_pred_traj=None
                    )
                )

            if self.with_map:
                map_thres = 0.7
                score_list = outputs[k]['pts_bbox']['score_list'].cpu().numpy().transpose([1, 2, 0])
                predicted_map_seg = outputs[k]['pts_bbox']['lane_score'].cpu().numpy().transpose([1, 2, 0])
                predicted_map_seg[..., -1] = score_list[..., -1]
                predicted_map_seg = (predicted_map_seg > map_thres) * 1.0
                predicted_map_seg = predicted_map_seg[::-1, :, :]
            else:
                predicted_map_seg = None

            if self.with_planning:
                # Add the SDC planning result
                bboxes = outputs[k]['sdc_boxes_3d']
                scores = outputs[k]['sdc_scores_3d']
                labels = 0

                track_scores = scores.cpu().detach().numpy()
                track_labels = labels
                track_centers = bboxes.gravity_center.cpu().detach().numpy()
                track_dims = bboxes.dims.cpu().detach().numpy()
                track_yaw = bboxes.yaw.cpu().detach().numpy()
                track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

                if self.show_command:
                    command = outputs[k]['command'][0].cpu().detach().numpy()
                else:
                    command = None
                planning_agent = AgentPredictionData(
                    track_scores[0],
                    track_labels,
                    track_centers[0],
                    track_dims[0],
                    track_yaw[0],
                    track_velocity[0],
                    outputs[k]['planning_traj'][0].cpu().detach().numpy(),
                    1,
                    pred_track_id=-1,
                    pred_occ_map=None,
                    past_pred_traj=None,
                    is_sdc=True,
                    command=command,
                )
                predicted_agent_list.append(planning_agent)
            else:
                planning_agent = None

            prediction_dict[token] = dict(predicted_agent_list=predicted_agent_list,
                                          predicted_map_seg=predicted_map_seg,
                                          predicted_planning=planning_agent)
        return prediction_dict

    def get_gt_boxes(self, sample_token):
        # Retrieve GT boxes in global frame
        sample_record = self.nusc.get('sample', sample_token)
        sd_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        _, boxes, _ = self.nusc.get_sample_data(sd_record['token'])
        gt_boxes = []
        for box in boxes:
            center = box.center
            # Reorder wlh to length, width, height for consistency
            dims = np.array([box.wlh[1], box.wlh[0], box.wlh[2]])
            yaw = box.orientation.yaw_pitch_roll[0]
            gt_boxes.append(dict(
                center=center,
                dims=dims,
                yaw=yaw,
                box=box
            ))
        return gt_boxes

    def draw_3d_box_on_bev(self, ax, center, dims, yaw, color='green'):
        corners_3d = box_corners_3d(center, dims, yaw)
        box2d = bev_box_2d(corners_3d)
        xs = np.append(box2d[:,0], box2d[0,0])
        ys = np.append(box2d[:,1], box2d[0,1])
        ax.plot(xs, ys, color=color, linewidth=2)

    def visualize_bev(self, sample_token, out_filename):
        self.bev_render.reset_canvas(dx=1, dy=1)
        self.bev_render.set_plot_cfg()

        # Retrieve the current figure and axes
        fig = plt.gcf()
        ax = plt.gca()

        if self.show_lidar:
            self.bev_render.show_lidar_data(sample_token, self.nusc)

        gt_boxes = self.get_gt_boxes(sample_token)
        pred_agents = self.predictions[sample_token]['predicted_agent_list']

        # Draw GT boxes in green
        if self.show_gt_boxes:
            for gt_b in gt_boxes:
                self.draw_3d_box_on_bev(ax, gt_b['center'], gt_b['dims'], gt_b['yaw'], color='green')

        # Draw predicted boxes in red
        if self.with_pred_box:
            for p in pred_agents:
                self.draw_3d_box_on_bev(ax, p.center, p.size, p.yaw, color='red')

        if self.with_pred_traj:
            self.bev_render.render_pred_traj(pred_agents)
        if self.with_map:
            self.bev_render.render_pred_map_data(
                self.predictions[sample_token]['predicted_map_seg'])
        if self.with_occ_map:
            self.bev_render.render_occ_map_data(pred_agents)

        # Draw SDC box in blue if planning is enabled
        if self.with_planning and self.predictions[sample_token]['predicted_planning'] is not None:
            planning_agent = self.predictions[sample_token]['predicted_planning']
            self.draw_3d_box_on_bev(ax, planning_agent.center, planning_agent.size, planning_agent.yaw, color='blue')
            self.bev_render.render_planning_data(planning_agent, show_command=self.show_command)

        if self.show_hd_map:
            self.bev_render.render_hd_map(
                self.nusc, self.nusc_maps, sample_token)
        if self.show_sdc_car:
            self.bev_render.render_sdc_car()
        if self.show_legend:
            self.bev_render.render_legend()

        self.bev_render.save_fig(out_filename + '.jpg')

        # Compute IoU between matched pairs of GT and predicted boxes
        pred_boxes = []
        for p in pred_agents:
            pred_boxes.append(dict(
                center=p.center,
                dims=p.size,
                yaw=p.yaw
            ))

        # Simple nearest-center matching for demonstration
        used_pred = set()
        for gt_b in gt_boxes:
            if len(pred_boxes) == 0:
                break
            dists = [np.linalg.norm(gt_b['center'] - pb['center']) for pb in pred_boxes]
            min_idx = np.argmin(dists)
            if min_idx in used_pred:
                continue
            used_pred.add(min_idx)
            iou_3d, iou_bev = box_iou_3d(gt_b, pred_boxes[min_idx])
            self.all_3d_ious.append(iou_3d)
            self.all_bev_ious.append(iou_bev)

    def visualize_cam(self, sample_token, out_filename):
        self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
        self.cam_render.render_image_data(sample_token, self.nusc)
        self.cam_render.render_pred_track_bbox(
            self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc)
        self.cam_render.render_pred_traj(
            self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc, render_sdc=self.with_planning)
        self.cam_render.save_fig(out_filename + '_cam.jpg')

    def combine(self, out_filename):
        bev_image = cv2.imread(out_filename + '.jpg')
        cam_image = cv2.imread(out_filename + '_cam.jpg')
        merge_image = cv2.hconcat([cam_image, bev_image])
        cv2.imwrite(out_filename + '.jpg', merge_image)
        os.remove(out_filename + '_cam.jpg')

    def to_video(self, folder_path, out_path, fps=4, downsample=1):
        imgs_path = glob.glob(os.path.join(folder_path, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        size = None
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height // downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def main(args):
    render_cfg = dict(
        with_occ_map=False,
        with_map=False,
        with_planning=True,
        with_pred_box=True,
        with_pred_traj=True,
        show_gt_boxes=True,
        show_lidar=False,
        show_command=True,
        show_hd_map=False,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False
    )

    viser = Visualizer(version='v1.0-mini', predroot=args.predroot, dataroot='data/nuscenes', **render_cfg)

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    val_splits = splits.val

    scene_token_to_name = dict()
    for i in range(len(viser.nusc.scene)):
        scene_token_to_name[viser.nusc.scene[i]['token']] = viser.nusc.scene[i]['name']

    for i in range(len(viser.nusc.sample)):
        sample_token = viser.nusc.sample[i]['token']
        scene_token = viser.nusc.sample[i]['scene_token']

        if scene_token_to_name[scene_token] not in val_splits:
            continue

        if sample_token not in viser.token_set:
            print(i, sample_token, 'not in prediction pkl!')
            continue

        viser.visualize_bev(sample_token, os.path.join(args.out_folder, str(i).zfill(3)))

        if args.project_to_cam:
            viser.visualize_cam(sample_token, os.path.join(args.out_folder, str(i).zfill(3)))
            viser.combine(os.path.join(args.out_folder, str(i).zfill(3)))

    # Compute average IoUs after processing all samples
    if len(viser.all_3d_ious) > 0:
        avg_3d_iou = np.mean(viser.all_3d_ious)
        avg_bev_iou = np.mean(viser.all_bev_ious)
    else:
        avg_3d_iou = 0.0
        avg_bev_iou = 0.0

    print("Average 3D IoU:", avg_3d_iou)
    print("Average BEV IoU:", avg_bev_iou)

    viser.to_video(args.out_folder, args.demo_video, fps=4, downsample=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predroot', default='results.pkl', help='Path to results.pkl')
    parser.add_argument('--out_folder', default='viz_output/', help='Output folder path')
    parser.add_argument('--demo_video', default='mini_val_final.avi', help='Demo video name')
    parser.add_argument('--project_to_cam', action='store_true', help='Project to cam')
    args = parser.parse_args()
    main(args)
