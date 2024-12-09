import cv2
import torch
import argparse
import os
import glob
import numpy as np
import mmcv
import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from shapely.geometry import Polygon
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils import splits
from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap
from PIL import Image
from tools.analysis_tools.visualize.utils import AgentPredictionData, color_mapping
from tools.analysis_tools.visualize.render.base_render import BaseRender
from tools.analysis_tools.visualize.render.bev_render import BEVRender
from tools.analysis_tools.visualize.render.cam_render import CameraRender


def box_corners_3d(center, dims, yaw):
    l, w, h = dims
    x, y, z = center
    corners = np.array([
        [ l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2,  h/2],
        [ l/2, -w/2, -h/2],
        [-l/2,  w/2,  h/2],
        [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2],
        [-l/2, -w/2, -h/2]
    ])
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = (rot @ corners.T).T + np.array([x, y, z])
    return corners

def bev_polygon_2d(corners_3d):
    # Extract the polygon (rotated) from the top face of the 3D box
    top_face_indices = [0, 2, 6, 4]
    poly_points = corners_3d[top_face_indices, :2]
    return poly_points

def box_iou_bev(poly1_points, poly2_points):
    poly1 = Polygon(poly1_points)
    poly2 = Polygon(poly2_points)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def box_iou_3d(gt_box, pred_box, debug=False):
    gt_corners = box_corners_3d(gt_box['center'], gt_box['dims'], gt_box['yaw'])
    pd_corners = box_corners_3d(pred_box['center'], pred_box['dims'], pred_box['yaw'])

    gt_poly_pts = bev_polygon_2d(gt_corners)
    pd_poly_pts = bev_polygon_2d(pd_corners)

    poly_gt = Polygon(gt_poly_pts)
    poly_pd = Polygon(pd_poly_pts)

    if not poly_gt.is_valid or not poly_pd.is_valid:
        if debug:
            print("Invalid polygon detected. Returning IoU = 0.0")
        return 0.0, 0.0

    inter_area = poly_gt.intersection(poly_pd).area
    union_area = poly_gt.union(poly_pd).area
    bev_iou = inter_area / union_area if union_area > 0 else 0.0

    gt_zmin, gt_zmax = np.min(gt_corners[:,2]), np.max(gt_corners[:,2])
    pd_zmin, pd_zmax = np.min(pd_corners[:,2]), np.max(pd_corners[:,2])
    inter_h = max(0, min(gt_zmax, pd_zmax) - max(gt_zmin, pd_zmin))

    if inter_h <= 0:
        if debug:
            print(f"No vertical overlap: inter_h={inter_h}, 3D IoU=0.0, BEV IoU={bev_iou:.4f}")
        return 0.0, bev_iou

    inter_vol = inter_area * inter_h
    gt_vol = np.prod(gt_box['dims'])
    pd_vol = np.prod(pred_box['dims'])
    union_vol = gt_vol + pd_vol - inter_vol
    iou_3d = inter_vol / union_vol if union_vol > 0 else 0.0

    if debug:
        print("=== IoU Debug Info ===")
        print(f"GT center: {gt_box['center']}, dims: {gt_box['dims']}, yaw: {gt_box['yaw']}")
        print(f"Pred center: {pred_box['center']}, dims: {pred_box['dims']}, yaw: {pred_box['yaw']}")
        print(f"GT volume: {gt_vol:.4f}, Pred volume: {pd_vol:.4f}")
        print(f"Intersection area (BEV): {inter_area:.4f}, Intersection height: {inter_h:.4f}")
        print(f"Intersection volume: {inter_vol:.4f}")
        print(f"Union volume: {union_vol:.4f}")
        print(f"3D IoU: {iou_3d:.4f}, BEV IoU: {bev_iou:.4f}")
        print("======================\n")

    return iou_3d, bev_iou


class Visualizer:
    """
    Visualizer class that plots ground truth boxes in green, predicted boxes in red,
    and computes average 3D and BEV IoU. Additionally, creates debugging plots comparing
    BEV bounding boxes vs ground truth boxes.
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
        top_face_indices = [0, 2, 6, 4]
        box2d = corners_3d[top_face_indices, :2]
        xs = np.append(box2d[:,0], box2d[0,0])
        ys = np.append(box2d[:,1], box2d[0,1])
        ax.plot(xs, ys, color=color, linewidth=2)

    def visualize_debugging_bev(self, ax, gt_boxes, pred_boxes, matched_pairs, ious):
        """
        Creates an additional BEV plot for debugging by overlaying GT and predicted boxes,
        connecting matched pairs, and annotating IoU values.
        """
        # Plot Ground Truth Boxes in Green
        for gt in gt_boxes:
            self.draw_3d_box_on_bev(ax, gt['center'], gt['dims'], gt['yaw'], color='green')

        # Plot Predicted Boxes in Red
        for pred in pred_boxes:
            self.draw_3d_box_on_bev(ax, pred['center'], pred['dims'], pred['yaw'], color='red')

        # Connect Matched Pairs and Annotate IoU
        for idx, (gt_idx, pred_idx) in enumerate(matched_pairs):
            gt = gt_boxes[gt_idx]
            pred = pred_boxes[pred_idx]
            iou = ious[idx]

            # Calculate centers for connecting lines
            gt_center = gt['center'][:2]
            pred_center = pred['center'][:2]

            # Draw a line between GT and Predicted centers
            ax.plot(
                [gt_center[0], pred_center[0]],
                [gt_center[1], pred_center[1]],
                color='yellow',
                linestyle='--',
                linewidth=1
            )

            # Annotate IoU
            mid_point = (gt_center + pred_center) / 2
            ax.text(
                mid_point[0],
                mid_point[1],
                f"{iou:.2f}",
                color='yellow',
                fontsize=8,
                backgroundcolor='black'
            )

    def visualize_bev(self, sample_token, out_filename):
        self.bev_render.reset_canvas(dx=1, dy=1)
        self.bev_render.set_plot_cfg()

        fig, ax = plt.subplots(figsize=(10, 10))  # Increased figure size for better visibility

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
                # Updated attributes to match AgentPredictionData class definition
                self.draw_3d_box_on_bev(ax, p.pred_center, p.pred_dim, p.pred_yaw, color='red')

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
            self.draw_3d_box_on_bev(ax, planning_agent.pred_center, planning_agent.pred_dim, planning_agent.pred_yaw, color='blue')
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
                center=p.pred_center,
                dims=p.pred_dim,
                yaw=p.pred_yaw
            ))

        # Simple nearest-center matching for demonstration
        used_pred = set()
        matched_pairs = []
        ious = []
        debug = False  # Set to True if you want detailed debug info

        for gt_idx, gt_b in enumerate(gt_boxes):
            if len(pred_boxes) == 0:
                break
            dists = [np.linalg.norm(gt_b['center'] - pb['center']) for pb in pred_boxes]
            min_idx = np.argmin(dists)
            if min_idx in used_pred:
                continue
            used_pred.add(min_idx)
            iou_3d, iou_bev = box_iou_3d(gt_b, pred_boxes[min_idx], debug=debug)
            self.all_3d_ious.append(iou_3d)
            self.all_bev_ious.append(iou_bev)
            matched_pairs.append((gt_idx, min_idx))
            ious.append(iou_3d)  # You can choose to annotate with 3D or BEV IoU

        # Create Debugging Plot
        debug_out_filename = out_filename + '_debug'
        debug_fig, debug_ax = plt.subplots(figsize=(10, 10))

        # Configure the debug_ax independently using margin
        debug_ax.set_xlim([-self.bev_render.margin, self.bev_render.margin])
        debug_ax.set_ylim([-self.bev_render.margin, self.bev_render.margin])
        debug_ax.set_xlabel('X (meters)')
        debug_ax.set_ylabel('Y (meters)')
        debug_ax.set_title('BEV Debugging Plot: GT vs Predicted Boxes')
        debug_ax.grid(True)

        # Call the debugging visualization
        self.visualize_debugging_bev(debug_ax, gt_boxes, pred_boxes, matched_pairs, ious)

        # Save the debugging plot
        debug_fig.savefig(debug_out_filename + '.jpg')
        plt.close(debug_fig)  # Close the figure to free memory

        # Close the main plot
        plt.close(fig)

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


class BEVRender(BaseRender):
    """
    Render class for BEV
    """

    def __init__(self,
                 figsize=(20, 20),
                 margin: float = 50,
                 view: np.ndarray = np.eye(4),
                 show_gt_boxes=False):
        super(BEVRender, self).__init__(figsize)
        self.margin = margin
        self.view = view
        self.show_gt_boxes = show_gt_boxes

    def set_plot_cfg(self):
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])
        self.axes.set_aspect('equal')
        self.axes.grid(False)

    def render_sample_data(self, canvas, sample_token):
        pass

    def render_anno_data(
            self,
            sample_token,
            nusc,
            predict_helper):
        sample_record = nusc.get('sample', sample_token)
        assert 'LIDAR_TOP' in sample_record['data'].keys(
        ), 'Error: No LIDAR_TOP in data, unable to render.'
        lidar_record = sample_record['data']['LIDAR_TOP']
        data_path, boxes, _ = nusc.get_sample_data(
            lidar_record, selected_anntokens=sample_record['anns'])
        for box in boxes:
            instance_token = nusc.get('sample_annotation', box.token)[
                'instance_token']
            future_xy_local = predict_helper.get_future_for_agent(
                instance_token, sample_token, seconds=6, in_agent_frame=True)
            if future_xy_local.shape[0] > 0:
                trans = box.center
                rot = Quaternion(matrix=box.rotation_matrix)
                future_xy = convert_local_coords_to_global(
                    future_xy_local, trans, rot)
                future_xy = np.concatenate(
                    [trans[None, :2], future_xy], axis=0)
                c = np.array([0, 0.8, 0])
                box.render(self.axes, view=self.view, colors=(c, c, c))
                self._render_traj(future_xy, line_color=c, dot_color=(0, 0, 0))
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])

    def show_lidar_data(
            self,
            sample_token,
            nusc):
        sample_record = nusc.get('sample', sample_token)
        assert 'LIDAR_TOP' in sample_record['data'].keys(
        ), 'Error: No LIDAR_TOP in data, unable to render.'
        lidar_record = sample_record['data']['LIDAR_TOP']
        data_path, boxes, _ = nusc.get_sample_data(
            lidar_record, selected_anntokens=sample_record['anns'])
        LidarPointCloud.from_file(data_path).render_height(
            self.axes, view=self.view)
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])
        self.axes.axis('off')
        self.axes.set_aspect('equal')

    def render_pred_box_data(self, agent_prediction_list):
        for pred_agent in agent_prediction_list:
            c = np.array([0, 1, 0])
            if hasattr(pred_agent, 'pred_track_id') and pred_agent.pred_track_id is not None:
                tr_id = pred_agent.pred_track_id
                c = color_mapping[tr_id % len(color_mapping)]
            pred_agent.nusc_box.render(
                axis=self.axes, view=self.view, colors=(c, c, c))
            if pred_agent.is_sdc:
                c = np.array([1, 0, 0])
                pred_agent.nusc_box.render(
                    axis=self.axes, view=self.view, colors=(c, c, c))

    def render_pred_traj(self, agent_prediction_list, top_k=3):
        for pred_agent in agent_prediction_list:
            if pred_agent.is_sdc:
                continue
            sorted_ind = np.argsort(pred_agent.pred_traj_score)[
                ::-1]  # from high to low
            num_modes = len(sorted_ind)
            sorted_traj = pred_agent.pred_traj[sorted_ind, :, :2]
            sorted_score = pred_agent.pred_traj_score[sorted_ind]
            # norm_score = np.sum(np.exp(sorted_score))
            norm_score = np.exp(sorted_score[0])

            sorted_traj = np.concatenate(
                [np.zeros((num_modes, 1, 2)), sorted_traj], axis=1)
            trans = pred_agent.pred_center
            rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi/2)
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if pred_agent.pred_label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25
            # print(sorted_score)
            for i in range(top_k-1, -1, -1):
                viz_traj = sorted_traj[i, :, :2]
                viz_traj = convert_local_coords_to_global(viz_traj, trans, rot)
                traj_score = np.exp(sorted_score[i])/norm_score
                # traj_score = [1.0, 0.01, 0.01, 0.01, 0.01, 0.01][i]
                self._render_traj(viz_traj, traj_score=traj_score,
                                  colormap='winter', dot_size=dot_size)

    def render_pred_map_data(self, predicted_map_seg):
        # rendered_map = map_color_dict
        # divider, crossing, contour
        map_color_dict = np.array(
            [(204, 128, 0), (102, 255, 102), (102, 255, 102)])
        rendered_map = map_color_dict[predicted_map_seg.argmax(
            -1).reshape(-1)].reshape(200, 200, -1)
        bg_mask = predicted_map_seg.sum(-1) == 0
        rendered_map[bg_mask, :] = 255
        self.axes.imshow(rendered_map, alpha=0.6,
                         interpolation='nearest', extent=(-51.2, 51.2, -51.2, 51.2))

    def render_occ_map_data(self, agent_list):
        rendered_map = np.ones((200, 200, 3))
        rendered_map_hsv = matplotlib.colors.rgb_to_hsv(rendered_map)
        occ_prob_map = np.zeros((200, 200))
        for i in range(len(agent_list)):
            pred_agent = agent_list[i]
            if pred_agent.pred_occ_map is None:
                continue
            if hasattr(pred_agent, 'pred_track_id') and pred_agent.pred_track_id is not None:
                tr_id = pred_agent.pred_track_id
                c = color_mapping[tr_id % len(color_mapping)]
            pred_occ_map = pred_agent.pred_occ_map.max(0)
            update_mask = pred_occ_map > occ_prob_map
            occ_prob_map[update_mask] = pred_occ_map[update_mask]
            pred_occ_map *= update_mask
            hsv_c = matplotlib.colors.rgb_to_hsv(c)
            rendered_map_hsv[pred_occ_map > 0.1] = (
                np.ones((200, 200, 1)) * hsv_c)[pred_occ_map > 0.1]
            max_prob = pred_occ_map.max()
            renorm_pred_occ_map = (pred_occ_map - max_prob) * 0.7 + 1
            sat_map = (renorm_pred_occ_map * hsv_c[1])
            rendered_map_hsv[pred_occ_map > 0.1,
                             1] = sat_map[pred_occ_map > 0.1]
            rendered_map = matplotlib.colors.hsv_to_rgb(rendered_map_hsv)
        self.axes.imshow(rendered_map, alpha=0.8,
                         interpolation='nearest', extent=(-50, 50, -50, 50))

    def render_occ_map_data_time(self, agent_list, t):
        rendered_map = np.ones((200, 200, 3))
        rendered_map_hsv = matplotlib.colors.rgb_to_hsv(rendered_map)
        occ_prob_map = np.zeros((200, 200))
        for i in range(len(agent_list)):
            pred_agent = agent_list[i]
            if pred_agent.pred_occ_map is None:
                continue
            if hasattr(pred_agent, 'pred_track_id') and pred_agent.pred_track_id is not None:
                tr_id = pred_agent.pred_track_id
                c = color_mapping[tr_id % len(color_mapping)]
            pred_occ_map = pred_agent.pred_occ_map[t]
            update_mask = pred_occ_map > occ_prob_map
            occ_prob_map[update_mask] = pred_occ_map[update_mask]
            pred_occ_map *= update_mask
            hsv_c = matplotlib.colors.rgb_to_hsv(c)
            rendered_map_hsv[pred_occ_map > 0.1] = (
                np.ones((200, 200, 1)) * hsv_c)[pred_occ_map > 0.1]
            max_prob = pred_occ_map.max()
            renorm_pred_occ_map = (pred_occ_map - max_prob) * 0.7 + 1
            sat_map = (renorm_pred_occ_map * hsv_c[1])
            rendered_map_hsv[pred_occ_map > 0.1,
                             1] = sat_map[pred_occ_map > 0.1]
            rendered_map = matplotlib.colors.hsv_to_rgb(rendered_map_hsv)
        self.axes.imshow(rendered_map, alpha=0.8,
                         interpolation='nearest', extent=(-50, 50, -50, 50))

    def render_planning_data(self, predicted_planning, show_command=False):
        planning_traj = predicted_planning.pred_traj
        planning_traj = np.concatenate(
            [np.zeros((1, 2)), planning_traj], axis=0)
        self._render_traj(planning_traj, colormap='autumn', dot_size=50)
        if show_command:
            self._render_command(predicted_planning.command)

    def render_planning_attn_mask(self, predicted_planning):
        planning_attn_mask = predicted_planning.attn_mask
        planning_attn_mask = planning_attn_mask/planning_attn_mask.max()
        cmap_name = 'plasma'
        self.axes.imshow(planning_attn_mask, alpha=0.8, interpolation='nearest', extent=(
            -51.2, 51.2, -51.2, 51.2), vmax=0.2, cmap=matplotlib.colormaps[cmap_name])

    def render_hd_map(self, nusc, nusc_maps, sample_token):
        sample_record = nusc.get('sample', sample_token)
        sd_rec = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        info = {
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'scene_token': sample_record['scene_token']
        }

        layer_names = ['road_divider', 'road_segment', 'lane_divider',
                       'lane',  'road_divider', 'traffic_light', 'ped_crossing']
        map_mask = obtain_map_info(nusc,
                                   nusc_maps,
                                   info,
                                   patch_size=(102.4, 102.4),
                                   canvas_size=(1024, 1024),
                                   layer_names=layer_names)
        map_mask = np.flip(map_mask, axis=1)
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))
        map_mask = map_mask[:, ::-1] > 0
        map_show = np.ones((1024, 1024, 3))
        map_show[map_mask[0], :] = np.array([1.00, 0.50, 0.31])
        map_show[map_mask[1], :] = np.array([159./255., 0.0, 1.0])
        self.axes.imshow(map_show, alpha=0.2, interpolation='nearest',
                         extent=(-51.2, 51.2, -51.2, 51.2))

    def _render_traj(self, future_traj, traj_score=1, colormap='winter', points_per_step=20, line_color=None, dot_color=None, dot_size=25):
        total_steps = (len(future_traj)-1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors*traj_score + \
            (1-traj_score)*np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps-1):
            unit_vec = future_traj[i//points_per_step +
                                   1] - future_traj[i//points_per_step]
            total_xy[i] = (i/points_per_step - i//points_per_step) * \
                unit_vec + future_traj[i//points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(
            total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)

    def _render_command(self, command):
        command_dict = ['TURN RIGHT', 'TURN LEFT', 'KEEP FORWARD']
        self.axes.text(-48, -45, command_dict[int(command)], fontsize=45)

    def render_sdc_car(self):
        sdc_car_png = cv2.imread('sources/sdc_car.png')
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))

    def render_legend(self):
        legend = cv2.imread('sources/legend.png')
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(23, 51.2, -50, -40))


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
