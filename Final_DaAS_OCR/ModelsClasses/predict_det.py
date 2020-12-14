import os
import sys
import cv2
import numpy as np
import math
import time
import paddle.fluid as fluid


#Relative files import
from utils.img_read_utility import initial_logger
from utils.constant import *
from utils import utility
from ModelsPreprocess import ModelsProcess,ModelsPostProcess



logger = initial_logger()

class TextDetector(object):
    def __init__(self):
        max_side_len = DET_MAX_SIDE_LEN
        self.det_algorithm = DET_ALGORITHM
        preprocess_params = {'max_side_len':max_side_len}
        postprocess_params = {}
        if self.det_algorithm == "DB":
            self.preprocess_op = ModelsProcess.DBProcessTest(preprocess_params)
            postprocess_params["thresh"] = DET_DB_THRESH
            postprocess_params["box_thresh"] = DET_DB_BOX_THRESH
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = DET_DB_UNCLIP_RATIO
            self.postprocess_op = ModelsPostProcess.DBPostProcess(postprocess_params)

            '''More algorithms are to be added into this file'''
        else:
            logger.info("Currently we only support DB algorithm please change your algo to DB in constant")
            sys.exit(0)
        if USE_PDSERVING is False:
            self.use_zero_copy_run = USE_ZERO_COPY_RUN
            self.predictor, self.input_tensor, self.output_tensors = utility.create_predictor(mode="det")

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        im, ratio_list = self.preprocess_op(img)
        if im is None:
            return None, 0
        im = im.copy()
        starttime = time.time()
        if self.use_zero_copy_run:
            self.input_tensor.copy_from_cpu(im)
            self.predictor.zero_copy_run()
        else:
            im = fluid.core.PaddleTensor(im)
            self.predictor.run([im])
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        outs_dict = {}
        outs_dict['maps'] = outputs[0]

        dt_boxes_list = self.postprocess_op(outs_dict, [ratio_list])
        dt_boxes = dt_boxes_list[0]
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse




