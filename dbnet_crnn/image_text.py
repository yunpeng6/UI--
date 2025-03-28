import cv2
import copy
import numpy as np
import dbnet_crnn.tools.utility as utility
from service.image_utils import get_center_pos
import dbnet_crnn.tools.predict_det as predict_det
import dbnet_crnn.tools.predict_rec as predict_rec
import os
import logging

# 添加日志配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def sorted_boxes(dt_boxes):
   """
   Sort text boxes in order from top to bottom, left to right
   args:
       dt_boxes(array):detected text boxes with shape [4, 2]
   return:
       sorted boxes(array) with shape [4, 2]
   """
   num_boxes = dt_boxes.shape[0]
   sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
   _boxes = list(sorted_boxes)

   for i in range(num_boxes - 1):
       if abs(_boxes[i+1][0][1] - _boxes[i][0][1]) < 10 and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
           tmp = _boxes[i]
           _boxes[i] = _boxes[i + 1]
           _boxes[i + 1] = tmp
   return _boxes

class ImageText:
    def __init__(self, args):
        model_path = '/Users/maoyan/Documents/vision-ui/dbnet_crnn/modelv1.1/det/'

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # # 检查模型文件完整性
        # required_files = ['batch_norm_47.b_0']  # 假设这是必需的文件之一
        # for file in required_files:
        #     file_path = os.path.join(model_path, file)
        #     if not os.path.exists(file_path):
        #         raise FileNotFoundError(f"Model file does not exist: {file_path}")

        # 添加日志输出，检查模型文件路径
        logging.debug(f"Model path: {model_path}")

        try:
            self.text_detector = predict_det.TextDetector(args, model_path=model_path)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
        self.text_recognizer = predict_rec.TextRecognizer(args, model_path='dbnet_crnn/modelv1.1/rec/')

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
        dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height),
                                      borderMode=cv2.BORDER_REPLICATE,
                                      flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def get_ocr(self, img, max_side_len):
        ori_im = img.copy()
        dt_boxes = self.text_detector(img, max_side_len)
        if dt_boxes is None:
            return None, None
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        rec_res = self.text_recognizer(img_crop_list)
        return dt_boxes, rec_res

    def get_text(self, img, max_side_len, score_thresh=0.6):
        result = []
        dt_boxes, rec_res = self.get_ocr(img, max_side_len)
        for roi_ocr in list(zip(dt_boxes, rec_res)):
            roi_score = roi_ocr[1][1]
            if roi_score > score_thresh:
                boxes = roi_ocr[0]
                result.append({
                    'pos': get_center_pos(roi_ocr[0]),
                    'text': roi_ocr[1][0],
                    'score': round(float(roi_score), 2),
                    'elem_det_region': [boxes[0][0], boxes[0][1], boxes[2][0], boxes[2][1]]
                })
        return result


# 假设args是一个已经定义好的参数对象或字典
args = utility.parse_args()  # 或者使用其他方式初始化args
image_text = ImageText(args)