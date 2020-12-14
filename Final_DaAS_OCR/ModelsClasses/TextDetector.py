import os
import numpy as np
import cv2
import onnxruntime
from utils.constant import *
from utils import imgproc
from build_files import craft_utils
from ModelsClasses import predict_det

class TextDetectorClass(object):
    def __init__(self):
        self.detection_session = onnxruntime.InferenceSession(DETECTION_MODEL)
        self.refiner_session = None
        if REFINE:
            # self.refiner_session = onnxruntime.InferenceSession(REFINER_MODEL)
            self.refiner_session = predict_det.TextDetector()

    def prepare_input_for_text_detection(self,image):
        x = imgproc.normalizeMeanVariance(image)
        x = x.transpose(2, 0, 1)
        x = np.expand_dims(x, 0)
        return x

    def adjustResultCoordinates(self,polys, ratio_w, ratio_h, ratio_net=2):
        if len(polys) > 0:
            polys = np.array(polys)
            for k in range(len(polys)):
                if polys[k] is not None:
                    polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
        return polys

    def get_score_text(self,word_boxes,score_text):
        # word_boxes = self.adjustResultCoordinates(word_boxes, self.ratio_w, self.ratio_h)
        word_boxes = np.array(word_boxes)
        modified_cords = word_boxes.copy()
        boxes_width = word_boxes[:, 2, 1] - word_boxes[:, 1, 1]
        modified_cords[:, 0, 1] = modified_cords[:, 0, 1] + (
                (boxes_width) * 0.25)
        modified_cords[:, 1, 1] = modified_cords[:, 1, 1] + (
                (boxes_width) * 0.25)
        modified_cords[:, 2, 1] = modified_cords[:, 2, 1] - (
                (boxes_width) * 0.25)
        modified_cords[:, 3, 1] = modified_cords[:, 3, 1] - (
                (boxes_width) * 0.25)
        # dummy_zeros = np.zeros(self.original_shape[0:2])
        dummy_zeros = np.zeros(score_text.shape)
        for line in modified_cords:
            ctr = np.array(line).reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(dummy_zeros, [ctr], -1, 1, -1)
        # score_text = cv2.resize(dummy_zeros, (score_text.shape[1], score_text.shape[0]))
        score_text = dummy_zeros.copy()
        return score_text

    def get_text_regions(self,image,ratio_w,ratio_h,original_shape):
        self.ratio_w = ratio_w
        self.ratio_h = ratio_h
        self.original_shape = original_shape
        input_blob = self.prepare_input_for_text_detection(image)
        detection_input = {self.detection_session.get_inputs()[0].name: input_blob}
        y, feature = self.detection_session.run(None, detection_input)
        score_text = y[0, :, :, 0]
        score_link = y[0, :, :, 1]
        link_boxes = None
        word_boxes, polys = craft_utils.getDetBoxes(score_text, score_link, TEXT_THRESHOLD, LINK_THRESHOLD,
                                                        LOW_TEXT, POLY)
        if self.refiner_session is not None:
            link_img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            link_boxes, elapse = self.refiner_session(link_img)
            score_text = self.get_score_text(word_boxes,score_text)
            # refiner_input = {self.refiner_session.get_inputs()[0].name: y, self.refiner_session.get_inputs()[1].name: feature}
            # y_refiner = self.refiner_session.run(None, refiner_input)
            # score_link = y_refiner[0][0, :, :, 0]
            modified_cords = link_boxes.copy()
            boxes_width = link_boxes[:, 2, 1] - link_boxes[:, 1, 1]
            modified_cords[:, 0, 1] = modified_cords[:, 0, 1] + (
                        (boxes_width) * 0.25)
            modified_cords[:, 1, 1] = modified_cords[:, 1, 1] + (
                    (boxes_width) * 0.25)
            modified_cords[:, 2, 1] = modified_cords[:, 2, 1] - (
                    (boxes_width) * 0.25)
            modified_cords[:, 3, 1] = modified_cords[:, 3, 1] - (
                    (boxes_width) * 0.25)
            dummy_zeros = np.zeros((link_img.shape[0], link_img.shape[1]))
            for line in modified_cords:
                ctr = np.array(line).reshape((-1, 1, 2)).astype(np.int32)
                cv2.drawContours(dummy_zeros, [ctr], -1, 1, -1)
            score_link = cv2.resize(dummy_zeros,(score_text.shape[1],score_text.shape[0]))
            link_boxes, polys = craft_utils.getDetBoxes(score_text, score_link, TEXT_THRESHOLD, LINK_THRESHOLD, LOW_TEXT,
                                                   POLY)
        return link_boxes,word_boxes
