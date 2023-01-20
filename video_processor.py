"""Classes to handle processing of video data."""

import cv2
import pathlib


class VideoProcessor:
    """VideoProcessor converts a video file into sequence of frames (images).

    Attributes:
        path_to_video_file: Path to video file.
        fps: Frame per seconds. Ex: 10 fps = we get 10 images per 
        second of video
        video_name: Name of video.
    """

    def __init__(self, path_to_video_file: str, fps: int):

        self.path_to_video_file = path_to_video_file
        self.fps = fps
        self.video_name = self._get_video_name()

    def _get_video_name(self) -> str:
        """Parses video name from path to video file.

        Returns:
            Name of video file.
        """
        video_file = self.path_to_video_file.split('/')[1]
        video_name = "".join(video_file).split(".")[0]
        return video_name

    def video_to_frames(self) -> None:
        """Converts video to sequence of frames (images) and 
        writes output to data folder.
        """
        vidcap = cv2.VideoCapture(self.path_to_video_file)
        success, image = vidcap.read()

        path_to_frame_data = f'data/{self.video_name}/frames'
        pathlib.Path(path_to_frame_data).mkdir(parents=True, exist_ok=True)

        count = 0
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * (1000 * 1 / self.fps)))
            cv2.imwrite(f'{path_to_frame_data}/frame_{count}.jpg', image)
            success, image = vidcap.read()
            count += 1