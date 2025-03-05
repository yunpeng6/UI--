import os
from paddle.fluid.libpaddle import AnalysisConfig  # 修改为 paddle.fluid.libpaddle.AnalysisConfig
import paddle.fluid.libpaddle as paddle_infer  # 修改为 paddle.fluid.libpaddle

def parse_args():
    params = dict()

    # prediction engine
    params["r_optim"] = True
    params["use_tensorrt"] = False
    params["gpu_mem"] = 8000
    params["use_gpu"] = False  # 添加 use_gpu 默认值
    params["gpu_id"] = 0  # 添加 gpu_id 默认值

    # text detector
    params["det_algorithm"] = 'DB'
    params["det_max_side_len"] = 1500

    # DB Net
    params["det_db_thresh"] = 0.3
    params["det_db_box_thresh"] = 0.5
    params["det_db_unclip_ratio"] = 2.0

    # text recognizer
    params["rec_algorithm"] = 'CRNN'
    params["rec_image_shape"] = "3, 32, 320"
    params["rec_char_type"] = 'ch'
    params["rec_batch_num"] = 30
    params["max_text_length"] = 25
    params["rec_char_dict_path"] = "dbnet_crnn/ppocr/utils/keys.txt"
    params["use_space_char"] = True
    params["enable_mkldnn"] = False
    params["use_zero_copy_run"] = False
    return params


def create_predictor(args, mode="det", model_path="", **kwargs):
    """
    Create paddle predictor
    """
    if mode == "det":
        config = AnalysisConfig()  # 使用 paddle.fluid.libpaddle.AnalysisConfig
        config.set_model(model_path)
        config.switch_use_feed_fetch_ops(False)
        config.switch_specify_input_names(True)
        if args['enable_mkldnn']:
            config.enable_mkldnn()
        if args['use_gpu']:
            config.enable_gpu(args['gpu_mem'], args['gpu_id'])
        else:
            config.disable_gpu()
        predictor = paddle_infer.create_predictor(config)  # 使用 paddle.fluid.libpaddle.create_predictor
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        output_names = predictor.get_output_names()
        output_tensors = []
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
        return predictor, input_tensor, output_tensors
    else:
        raise ValueError("mode should be 'det' or 'rec'")
