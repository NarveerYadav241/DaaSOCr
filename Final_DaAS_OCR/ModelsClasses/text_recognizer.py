import numpy as np
import os
import cv2
import pickle
import tensorflow as tf
from utils.constant import OP_DICT_FILE,MODEL_PATH

class word_recognizer(object):
    def __init__(self):
        with open(OP_DICT_FILE, "rb") as mydict:
            op_dict = pickle.load(mydict)
        with open(MODEL_PATH, 'rb') as f:
            output_graph_def = tf.GraphDef()
            output_graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(output_graph_def, name='', op_dict=op_dict)
                self.sess = tf.Session(graph=graph)
                self.sess.run(graph.get_operation_by_name('init_all_tables'))
                self.graph = graph


    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def sort_contours(self,ctrs):
        # sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
        mean_x = (np.min(np.array(ctrs)[:, :, 0],axis=1) + np.max(np.array(ctrs)[:, :, 0],axis=1)) / 2
        sorted_mean = np.argsort(mean_x)
        sorted_ctrs = np.array(ctrs)[sorted_mean]
        return sorted_ctrs

    def recognize_text(self,word_boxes,image):
        doc_text = []
        doc_words = []
        words_cords = []
        for block in word_boxes:
            if len(block)!=0:
                block = self.sort_contours(block)
                line_text = []
                for word in block:
                    cropped_image = self.get_rotate_crop_image(image, word)
                    jpeg_image = tf.image.encode_jpeg(cropped_image, format='', quality=100, progressive=False,
                                                      optimize_size=False,
                                                      chroma_downsampling=True, density_unit='in', x_density=300,
                                                      y_density=300,
                                                      xmp_metadata='', name=None)
                    with tf.Session() as jpsess:
                        jpeg_string = jpsess.run(jpeg_image)

                    try:
                        result = self.sess.run(self.graph.get_tensor_by_name("strided_slice:0"),
                                               feed_dict={self.graph.get_tensor_by_name(
                                                   "Placeholder:0"): jpeg_string})
                        recognized_text = result.decode('utf-8')
                        line_text.append(recognized_text)
                        doc_words.append(recognized_text)
                        words_cords.append(word)
                    except Exception as e:
                        print("Error in Recognition as: ", e)
                doc_text.append(" ".join(line_text))
        print(doc_text)
        return doc_text,doc_words,words_cords


    # def recognize_text(self,word_boxes,image):
    #     doc_text = []
    #     for block in word_boxes:
    #         for line in block:
    #             block_text = []
    #             for word in line:
    #                 line_text = []
    #                 for c in word:
    #                     cropped_image = self.get_rotate_crop_image(image,c)
    #                     # cv2.imshow("cropped",cropped_image)
    #                     # cv2.waitKey(0)
    #                     # cv2.destroyAllWindows()
    #                     jpeg_image = tf.image.encode_jpeg(cropped_image, format='', quality=100, progressive=False,
    #                                                       optimize_size=False,
    #                                                       chroma_downsampling=True, density_unit='in', x_density=300,
    #                                                       y_density=300,
    #                                                       xmp_metadata='', name=None)
    #                     with tf.Session() as jpsess:
    #                         jpeg_string = jpsess.run(jpeg_image)
    #
    #                     try:
    #                         result = self.sess.run(self.graph.get_tensor_by_name("strided_slice:0"),
    #                                                          feed_dict={self.graph.get_tensor_by_name(
    #                                                              "Placeholder:0"): jpeg_string})
    #                         recognized_text = result.decode('utf-8')
    #                         line_text.append(recognized_text)
    #                     except Exception as e:
    #                         print("Error in Recognition as: ",e)
    #                 block_text.append(" ".join(line_text))
    #             doc_text.append(block_text)
    #     print(doc_text)
    #     return doc_text

