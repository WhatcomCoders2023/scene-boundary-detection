"""Classes to handle processing of video data."""

import cv2
import pathlib

from typing import Mapping


class VideoProcessor:
    """VideoProcessor converts a video file into sequence of frames (images).

    Attributes:
        path_to_video_file: Path to video file.
        fps: Frame per seconds. Ex: 10 fps = we get 10 images per 
        second of video
        video_name: Name of video.
        frame_number_to_timestamp_milliseconds: Map of frame number to timestamp 
        in milliseconds.
    """

    def __init__(self, path_to_video_file: str, fps: int):

        self.path_to_video_file = path_to_video_file
        self.fps = fps
        self.video_name = self._get_video_name()
        self.frame_number_to_timestamp_milliseconds = self._get_frame_to_timestamp_milliseconds(
        )

    def _get_video_name(self) -> str:
        """Parses video name from path to video file.

        Returns:
            Name of video file.
        """
        video_file = self.path_to_video_file.split('/')[1]
        video_name = "".join(video_file).split(".")[0]
        return video_name

    def _get_frame_to_timestamp_milliseconds(self) -> Mapping[int, float]:
        """Creates a map of frame number to timestamp (in milliseconds).

        Number of frames is determined by duration of video * frame per second.
        This method calculate duration of video using the metadata. And multiplies
        duration by the passed fps rate.

        Returns:
            Map of frame number to timestamp
        """
        frame_number_to_timestamp = {}
        vidcap = cv2.VideoCapture(self.path_to_video_file)

        default_number_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        default_fps = vidcap.get(cv2.CAP_PROP_FPS)
        duration_of_video_in_seconds = int(default_number_of_frames /
                                           default_fps)

        adjusted_number_of_frames = int(duration_of_video_in_seconds * self.fps)
        for frame_number in range(adjusted_number_of_frames):
            timestamp = frame_number * (1000 * (1 / self.fps))
            frame_number_to_timestamp[frame_number] = timestamp
        return frame_number_to_timestamp

    def video_to_frames(self) -> None:
        """Converts video to sequence of frames (images) and 
        writes output to data folder.
        """
        vidcap = cv2.VideoCapture(self.path_to_video_file)
        success, image = vidcap.read()

        path_to_frame_data = f'data/{self.video_name}/frames'
        pathlib.Path(path_to_frame_data).mkdir(parents=True, exist_ok=True)

        frame_number = 0
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,
                       (frame_number * (1000 * (1 / self.fps))))
            cv2.imwrite(f'{path_to_frame_data}/frame_{frame_number}.jpg', image)
            success, image = vidcap.read()
            frame_number += 1


VideoProcessor('movies/nuclearFamily.mp4', 1).video_to_frames()