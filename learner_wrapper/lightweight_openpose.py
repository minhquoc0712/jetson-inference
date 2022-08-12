from opendr.perception.pose_estimation import LightweightOpenPoseLearner
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

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader



class LightweightOpenPoseLearnerWrapper(LightweightOpenPoseLearner):
    def __init__(
        self,
        backbone,
        device, 
        num_refinement_stages,
        mobilenet_use_stride,
        half_precision,
        *args,
        **kwargs):
        super(LightweightOpenPoseLearnerWrapper, self).__init__(
            backbone=backbone,
            device=device, num_refinement_stages=num_refinement_stages,
            mobilenet_use_stride=mobilenet_use_stride, half_precision=half_precision
        )