import os
import sys
import cv2
import numpy as np
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
#Relative file imports
from utils.constant import *
from utils.img_read_utility import initial_logger

logger = initial_logger()


def create_predictor(mode):
    model_dir = None
    if mode == "det":
        model_dir = REFINER_MODEL
    # elif mode == 'cls':
    #     model_dir = CLASS_PREDICTION_MODEL
    # else:
    #     model_dir = RECOGNITION_MODEL

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    model_file_path = model_dir + "/model"
    params_file_path = model_dir + "/params"
    if not os.path.exists(model_file_path):
        logger.info("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        logger.info("not find params file path {}".format(params_file_path))
        sys.exit(0)

    config = AnalysisConfig(model_file_path, params_file_path)

    if USE_GPU:
        config.enable_use_gpu(GPU_MEM, 0)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(6)
        if ENABLE_MKLDNN:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()

    # config.enable_memory_optim()
    config.disable_glog_info()

    if USE_ZERO_COPY_RUN:
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
    else:
        config.switch_use_feed_fetch_ops(True)

    predictor = create_paddle_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_tensor(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_tensor(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors
