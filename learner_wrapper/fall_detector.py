from typing_extensions import runtime
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.fall_detection import FallDetectorLearner
from opendr.engine.learners import Learner


# General imports
import onnxruntime as ort
import os
import ntpath
import shutil
import cv2
import torch
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import DataParallel
from tensorboardX import SummaryWriter
from torchvision import transforms
from urllib.request import urlretrieve

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from opendr.engine.data import Image
from opendr.engine.target import Pose
from opendr.engine.constants import OPENDR_SERVER_URL

# OpenDR lightweight_open_pose imports
from opendr.perception.pose_estimation.lightweight_open_pose.filtered_pose import FilteredPose
from opendr.perception.pose_estimation.lightweight_open_pose.utilities import track_poses
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.models.with_mobilenet import \
    PoseEstimationWithMobileNet
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.models.with_mobilenet_v2 import \
    PoseEstimationWithMobileNetV2
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.models.with_shufflenet import \
    PoseEstimationWithShuffleNet
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.get_parameters import \
    get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.load_state import \
    load_state  # , load_from_mobilenet
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.loss import l2_loss
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.keypoints import \
    extract_keypoints, group_keypoints
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.datasets.coco import CocoTrainDataset
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.datasets.coco import CocoValDataset
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.datasets.transformations import \
    ConvertKeypoints, Scale, Rotate, CropPad, Flip
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.val import \
    convert_to_coco_format, run_coco_eval, normalize, pad_width
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.scripts import \
    prepare_train_labels, make_val_subset


class FallDetectorLearnerWrapper(FallDetectorLearner):
    def __init__(self, pose_estimator, *args, **kwargs):
        super(FallDetectorLearnerWrapper, self).__init__(pose_estimator)

    def infer(self, img, mode):
        poses = self.pose_estimator_infer(img, mode)
        results = []
        for pose in poses:
            results.append(self.__naive_fall_detection(pose))

        if len(results) >= 1:
            return results

        return []

    def pose_estimator_infer(self, img, mode, upsample_ratio=4, track=True, smooth=True):

        tensor_img, scale, pad = self.preprocess(img)

        # Model forward pass.
        if mode == 'torch':
            if self.pose_estimator.model is None:
                raise UserWarning("No model is loaded, cannot run inference. Load a model first using load().")
            if self.pose_estimator.model_train_state:
                self.pose_estimator.model.eval()
                self.pose_estimator.model_train_state = False
            stages_output = self.pose_estimator.model(tensor_img)
            stage2_heatmaps = stages_output[-2]
            stage2_pafs = stages_output[-1]
        elif mode == 'onnx':
            stages_output = self.onnx_model(np.array(tensor_img.cpu()))
            stage2_heatmaps = torch.tensor(stages_output[-2])
            stage2_pafs = torch.tensor(stages_output[-1])
        else:
            raise RuntimeError('Not implemented')

        current_poses = self.postprocess(stage2_heatmaps, stage2_pafs, upsample_ratio, track, smooth, scale, pad)
       
        return current_poses

    def preprocess(self, img):
        img = img[0]  # Assume batch size is 1. 
        
        if not isinstance(img, Image):
            img = Image(img)

        # Bring image into the appropriate format for the implementation
        img = img.convert(format='channels_last', channel_order='bgr')

        height, width, _ = img.shape
        scale = self.pose_estimator.base_height / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, self.pose_estimator.img_mean, self.pose_estimator.img_scale)
        min_dims = [self.pose_estimator.base_height, max(scaled_img.shape[1], self.pose_estimator.base_height)]
        padded_img, pad = pad_width(scaled_img, self.pose_estimator.stride, self.pose_estimator.pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if "cuda" in self.pose_estimator.device:
            tensor_img = tensor_img.to(self.pose_estimator.device)
            if self.pose_estimator.half:
                tensor_img = tensor_img.half()

        return tensor_img, scale, pad

    def postprocess(self, stage2_heatmaps, stage2_pafs, upsample_ratio, track, smooth, scale, pad):
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        
        if self.pose_estimator.half:
            heatmaps = np.float32(heatmaps)
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        if self.pose_estimator.half:
            pafs = np.float32(pafs)
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        num_keypoints = 18
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.pose_estimator.stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.pose_estimator.stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            if smooth:
                pose = FilteredPose(pose_keypoints, pose_entries[n][18])
            else:
                pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(self.pose_estimator.previous_poses, current_poses, smooth=smooth)
            self.pose_estimator.previous_poses = current_poses
