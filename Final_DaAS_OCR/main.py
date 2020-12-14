import os
import numpy as np
import time
import cv2
from ModelsClasses import TextDetector
from ModelsClasses import text_recognizer
from tensorflow.contrib import rnn
from c_ops import ops
# from builders import model_builder
from utils import imgproc
from utils.constant import *
from build_files import craft_utils
from utils import file_utils

files_extensions = ['.jpeg','.jpg','.png']


def get_doc_info(lines_boxes,word_boxes,image):
    doc_info = []
    init_time = time.time()
    for line in lines_boxes:
        occupied_boxes = []
        line_image = np.zeros(image.shape[0:2])
        line_cnt = np.array(line).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(line_image,[line_cnt],-1,255,-1)
        words = []
        not_found_count = 0
        for i,box in enumerate(word_boxes):
            box_image = np.zeros(image.shape[0:2])
            box_cnt = np.array(box).reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(box_image,[box_cnt],-1,255,-1)
            intersection_img = cv2.bitwise_and(line_image,box_image)
            if intersection_img.any():
                conture_area = cv2.contourArea(box_cnt)
                contours, hierarchy = cv2.findContours(intersection_img.astype(np.uint8), cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    overlapping_area = cv2.contourArea(cnt)
                    ratio = overlapping_area/conture_area
                    if ratio >= CONTOUR_OVERLAPPING:
                        not_found_count = 0
                        words.append(box)
                        occupied_boxes.append(i)
                    else:
                        not_found_count = not_found_count + 1
            else:
                not_found_count = not_found_count + 1
            if not_found_count > NON_INTERSECT_COUNT:
                break
        word_boxes = np.delete(word_boxes,occupied_boxes,axis=0)
        doc_info.append(words)
    final_time = time.time()
    print("Words segregation took: ",final_time-init_time)
    return doc_info




# def  get_doc_info(line_boxes,word_boxes,image):
#     # word_mean_y = (word_boxes[:,2,1]+word_boxes[:,1,1])/2
#     # word_mean_x = (word_boxes[:,1,0]+word_boxes[:,0,0])/2
#     word_mean_y = (np.max(word_boxes[:,:,1],axis=1)+np.min(word_boxes[:,:,1],axis=1))/2
#     word_mean_x = (np.max(word_boxes[:,:,0],axis=1)+np.min(word_boxes[:,:,0],axis=1))/2
#     doc_info = []
#     for line in line_boxes:
#         min_y = np.min(line[:, 1], axis=0)
#         max_y = np.max(line[:,1],axis=0)
#         min_x = np.min(line[:,0],axis=0)
#         max_x = np.max(line[:,0],axis=0)
#         # original_image = image.copy()
#         # modified_image = image.copy()
#         block_info = []
#         # words_index_in_line = np.where((word_mean_y >= line[0][1])&(word_mean_y <= line[2][1])&(word_mean_x >= line[0][0])&(word_mean_x <= line[1][0]))
#         words_index_in_line = np.where((word_mean_y >= min_y)&(word_mean_y <= max_y)&(word_mean_x >= min_x)&(word_mean_x <= max_x))
#         words_cordinates_in_line = word_boxes[words_index_in_line]
#         words_mean_x_in_line = word_mean_x[words_index_in_line]
#         words_mean_y_in_line = word_mean_y[words_index_in_line]
#         sorted_y_index_in_line = np.argsort(words_mean_y_in_line)
#         words_sorted_with_y = words_cordinates_in_line[sorted_y_index_in_line]
#         words_mean_x_sorted_with_y = words_mean_x_in_line[sorted_y_index_in_line]
#         sorted_y_mean = words_mean_y_in_line[sorted_y_index_in_line]
#         average_words_width = np.average(words_sorted_with_y[:,2,1]-words_sorted_with_y[:,1,1])
#         permissible_y_shift = average_words_width
#         line_info = []
#         # for word in words_sorted_with_y:
#         #     cnt = np.array(word).reshape((-1, 1, 2)).astype(np.int32)
#         #     cv2.drawContours(original_image,[cnt],-1,(255,0,230),3)
#         # cv2.imshow("original",cv2.resize(original_image,None,None,fx=0.5,fy=0.5))
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#         while(sorted_y_mean.shape[0]!=0):
#             one_block_y = sorted_y_mean[0]
#             y_index_in_line = np.where((sorted_y_mean >= (one_block_y-permissible_y_shift))&(sorted_y_mean <= (one_block_y+permissible_y_shift)))
#             mean_x_in_line = words_mean_x_sorted_with_y[y_index_in_line]
#             final_words_sorted = words_sorted_with_y[y_index_in_line]
#             sorted_index_bounding_boxes = np.argsort(mean_x_in_line)
#             final_words = final_words_sorted[sorted_index_bounding_boxes]
#             sorted_y_mean = np.delete(sorted_y_mean, y_index_in_line,axis=0)
#             mean_x_in_line = np.delete(mean_x_in_line,y_index_in_line,axis=0)
#             final_words_sorted = np.delete(final_words_sorted,y_index_in_line,axis=0)
#             words_mean_x_sorted_with_y = np.delete(words_mean_x_sorted_with_y,y_index_in_line,axis=0)
#             words_sorted_with_y = np.delete(words_sorted_with_y,y_index_in_line,axis=0)
#             # for word in final_words:
#             #     cnt = np.array(word).reshape((-1, 1, 2)).astype(np.int32)
#             #     cv2.drawContours(modified_image, [cnt], -1, (255, 0, 230), -1)
#             # cv2.imshow("modified", cv2.resize(modified_image,None,None,fx=0.5,fy=0.5))
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()
#             line_info.append(final_words)
#         block_info.append(line_info)
#         doc_info.append(block_info)
#     return doc_info


def run_detection():
    text_detector_instance = TextDetector.TextDetectorClass()
    text_recognition = text_recognizer.word_recognizer()
    image_files = os.listdir(INPUT_IMAGES_DIR)

    for image_name in image_files:
        if os.path.splitext(image_name)[1] in files_extensions:
            image = imgproc.loadImage(os.path.join(INPUT_IMAGES_DIR,image_name))
            box_image = image.copy()
            img_start_time = time.time()
            img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, CANVAS_SIZE,
                                                                                  interpolation=cv2.INTER_LINEAR,
                                                                                  mag_ratio=MAG_RATIO)
            ratio_h = ratio_w = 1 / target_ratio
            # refiner_h_ratio = img_resized.shape[0]/image.shape[0]
            # refiner_w_ratio = img_resized.shape[1]/image.shape[1]
            line_boxes,word_boxes = text_detector_instance.get_text_regions(img_resized,ratio_w,ratio_h,(image.shape))
            word_boxes = craft_utils.adjustResultCoordinates(word_boxes, ratio_w, ratio_h)
            if line_boxes is not None:
                line_boxes = craft_utils.adjustResultCoordinates(line_boxes, ratio_w, ratio_h)

                # line_boxes = craft_utils.adjustRefinerCordinates(line_boxes,ratio_h,ratio_w)
                # word_boxes = line_boxes.copy()
                lines = line_boxes.copy()
            # polys = crft_utils.refine_polys(polys,boxes)
            doc_info = get_doc_info(line_boxes,word_boxes,image)
            doc_text,doc_words,words_cords = text_recognition.recognize_text(doc_info,image)
            end_time = time.time()
            print("elapsed time : {}s".format(end_time - img_start_time))
            if WRITE_LINES:
                file_utils.saveResult(image_name, image[:, :, ::-1], lines, dirname=RESULT_DIR)
            else:
                file_utils.saveResult(image_name, image[:, :, ::-1], word_boxes, dirname=RESULT_DIR)
            if WRITE_INFO:
                res_file = os.path.join(RESULT_DIR, "words_" + image_name.split(".")[0] + '.txt')
                with open(res_file, 'w') as f:
                    for i, box in enumerate(words_cords):
                        poly = np.array(box).astype(np.int32).reshape((-1))
                        strResult = ','.join([str(p) for p in poly]) + "," + doc_words[i] + '\r\n'
                        f.write(strResult)
                # res_file = os.path.join(RESULT_DIR, "lines_" + image_name.split(".")[0] + '.txt')
                # with open(res_file,'w') as line_file:
                #     for i, box in enumerate(lines):
                #         poly = np.array(box).astype(np.int32).reshape((-1))
                #         strResult = ','.join([str(p) for p in poly]) + "," + doc_text[i] + '\r\n'
                #         line_file.write(strResult)

# word_boxes[np.where((word_centroid_x>=one_line[0][0])&(word_centroid_x<=one_line[1][0])&(word_centroid_y>=one_line[0][1])&(word_centroid_y<=one_line[2][1]))]
# sorted_boxes = sorted(sorted_cords, key=lambda x: (x[0][1], x[0][0]))




# for block in doc_info:
#     for line in block:
#         color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
#         for word in line:
#             for c in word:
#                 ctr = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
#                 cv2.drawContours(word_image,[ctr],-1,color,2)




if __name__ == "__main__":
    run_detection()

