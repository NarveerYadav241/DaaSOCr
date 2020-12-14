import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sep = os.path.sep
DETECTION_MODEL = os.path.join(project_dir,"Models","textdetector.onnx")
# REFINER_MODEL = os.path.join(project_dir,"Models","Refiner.onnx")
REFINER_MODEL = os.path.join(project_dir,"Models","RefinerModel")
INPUT_IMAGES_DIR = os.path.join(project_dir,"Images")
RESULT_DIR = os.path.join(project_dir,"Result")
TEXT_THRESHOLD = 0.5
LOW_TEXT = 0.4
LINK_THRESHOLD = 0.4
USE_CUDA = False
CANVAS_SIZE = 1280
MAG_RATIO = 1.5
POLY = False
REFINE = True
# Y_VARIENCE = 10
###Refiner Parameters
USE_GPU = False
IR_OPTIM = True
USE_TENSORRT = False
GPU_MEM = 8000
##########Params for text detector#####################
DET_ALGORITHM = "DB"
DET_MAX_SIDE_LEN = 960.0
##################DB Params############################
DET_DB_THRESH = 0.3
DET_DB_BOX_THRESH = 0.5
DET_DB_UNCLIP_RATIO = 1.6
##################EAST Params##########################
DET_EAST_SCORE_THRESH = 0.8
DET_EAST_COVER_THRESH = 0.1
DET_EAST_NMS_THRESH = 0.2
##################SAST Params##########################
DET_SAST_SCORE_THRESH = 0.5
DET_SAST_NMS_THRESH = 0.2
DET_SAST_POLYGON = False
USE_ANGLE_CLS = True
CLS_IMAGE_SHAPE = "3, 48, 192"
LABEL_LIST = ['0', '180']
CLS_BATCH_NUM = 30
CLS_THRESH = 0.9
ENABLE_MKLDNN = False
USE_ZERO_COPY_RUN = False
USE_PDSERVING = False
####################Text Recognition Constants##############
MODEL_PATH = project_dir + sep + "Models" + sep + "Recognition_Model" + sep + "saved_model.pb"
OP_DICT_FILE = project_dir + sep + "Pickle_files" + sep + "op_dict"
WRITE_LINES = True
WRITE_INFO = True
################Box seperation parameters######################
CONTOUR_OVERLAPPING = 0.5
NON_INTERSECT_COUNT = 15