import cv2

from entities.VideoFrame import VideoFrame


class Video:
    def __init__(self, video_path: str):
        self.__video_capture = self.__open_video(video_path)

    @property
    def total_frames(self) -> int:
        """Retorna o número total de frames do vídeo."""
        return int(self.__video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def release(self):
        """Libera o objeto VideoCapture."""
        self.__video_capture.release()

    def read_frame(self, frame_idx: int) -> VideoFrame | None:
        self.__video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.__video_capture.read()

        if success:
            return VideoFrame(frame)
        
        return None
        
    def __open_video(self, video_path) -> cv2.VideoCapture:
        """Abre o vídeo e retorna o objeto VideoCapture ou None em caso de erro."""
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        return video_capture

