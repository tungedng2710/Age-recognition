import os
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7face.models.experimental import attempt_load
from yolov7face.utils.datasets import LoadStreams, LoadImages
from yolov7face.utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from yolov7face.utils.plots import colors, plot_one_box
from yolov7face.utils.torch_utils import select_device, load_classifier, time_synchronized

def detect(image):
    weights = "weights/yolov7-tiny.pth"