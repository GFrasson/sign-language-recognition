import cv2
import numpy as np


class VideoFrame:
    def __init__(self, frame: cv2.typing.MatLike):
        self.__frame = frame

    @property
    def frame(self):
        return self.__frame

    def resize(self, width: int, height: int):
        self.__frame = cv2.resize(self.__frame, (width, height))
        return self
    
    def bgr_to_rgb(self):
        self.__frame = cv2.cvtColor(self.__frame, cv2.COLOR_BGR2RGB)
        return self
    
    def flip(self, flip_code: int):
        self.__frame = cv2.flip(self.__frame, flip_code)
        return self
    
    def rotate(self, angle: float):
        height, width = self.__frame.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        self.__frame = cv2.warpAffine(self.__frame, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        return self
    
    def translate(self, translate_x: int, translate_y: int):
        height, width = self.__frame.shape[:2]
        translation_matrix = np.float32([
            [1, 0, translate_x],
            [0, 1, translate_y]
        ])
        self.__frame = cv2.warpAffine(
            self.__frame,
            translation_matrix,
            (width, height),
            borderMode=cv2.BORDER_REFLECT
        )
        return self

    def crop_centralize(self, crop_ratio: float):
        height, width = self.__frame.shape[:2]

        crop_height = int(height * crop_ratio)
        crop_width = int(width * crop_ratio)
        
        self.__frame = self.__frame[crop_height:height - crop_height, crop_width:width - crop_width]
        self.__frame = cv2.resize(self.__frame, (width, height))
        
        return self

    def change_brightness(self, brightness_factor: float):
        self.__frame = cv2.convertScaleAbs(self.__frame, alpha=brightness_factor, beta=0)
        return self

    def change_contrast(self, contrast_value: float):
        self.__frame = cv2.add(self.__frame, np.array([contrast_value]))
        return self
