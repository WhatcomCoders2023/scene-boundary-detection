# Class and methods for handling shot change detection in videos.
# Shot Detection Code repurposed from: https://github.com/HeliosZhao/Shot-Boundary-Detection

import cv2
import csv
import numpy as np
import pathlib
import csv

from typing import Sequence, Mapping, Tuple


class ShotsToFrames:
    """ShotsToFrames maps a video data to frame data.

    Attributes:
        shots_to_timestamp: Map of shot number to start and end timestamp.
        shots_to_frames: Map of shot number to list of frames.
        shots_to_frames: Path to people id in video's frame.
    """

    def __init__(
        self,
        path_to_shot_data: str,
        frames_to_timestamp: Mapping[int, float],
    ):
        self.frames_to_timestamp = frames_to_timestamp
        self.shots_to_timestamp = self._create_shots_to_timestamp(
            path_to_shot_data)
        self.shots_to_frames = self._create_shots_to_frames()

    def _create_shots_to_timestamp(
        self,
        path_to_shot_data: str,
    ) -> Mapping[int, Tuple[float, float]]:
        """Creates a map of shot number to its timestamp.

        Attributes:
            path_to_shot_data: Path to a video's shot changes data.
        
        Return:
            Map of shot number to timestamp, defined as a tuple 
            corresponding to (start_time, end_time).
        """
        shots_to_timestamp = {}
        with open(path_to_shot_data) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                shot_number = int(row[0])
                start_time, end_time = float(row[1]), float(row[2])
                shots_to_timestamp[shot_number] = (start_time, end_time)
        return shots_to_timestamp

    def _create_shots_to_frames(self) -> Mapping[int, Sequence[int]]:
        """Create a map of shot number to list of frames.

        Return:
            Map of shot number to list of frames in shot.
        """
        shots_to_frames = {}
        frame_number = 0
        for shot_number, (start_time,
                          end_time) in self.shots_to_timestamp.items():
            shots_to_frames[shot_number] = []
            while frame_number in self.frames_to_timestamp and end_time * 1000 > self.frames_to_timestamp[
                    frame_number]:
                shots_to_frames[shot_number].append(frame_number)
                frame_number += 1
        return shots_to_frames


class Frame:
    """class to hold information about each frame
    """

    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def getMAXdiff(self, list=[]):
        """
        find the max_diff_frame in the window
        """
        LIST = list[:]
        temp = LIST[0]
        for i in range(0, len(LIST)):
            if temp.diff > LIST[i].diff:
                continue
            else:
                temp = LIST[i]

        return temp

    def find_possible_frame(self, list_frames):
        """
        detect the possible frame
        """
        possible_frame = []
        window_frame = []
        window_size = 30
        m_suddenJudge = 3
        m_MinLengthOfShot = 8
        start_id_spot = []
        start_id_spot.append(0)
        end_id_spot = []

        length = len(list_frames)
        index = 0
        while (index < length):
            frame_item = list_frames[index]
            window_frame.append(frame_item)
            if len(window_frame) < window_size:
                index += 1
                if index == length - 1:
                    window_frame.append(list_frames[index])
                else:
                    continue

            # find the max_diff_frame
            max_diff_frame = self.getMAXdiff(window_frame)
            max_diff_id = max_diff_frame.id

            if len(possible_frame) == 0:
                possible_frame.append(max_diff_frame)
                continue
            last_max_frame = possible_frame[-1]
            """
            
            Check whether the difference of the selected frame is more than 3 times the average difference of the other frames in the window.
            
            """

            sum_start_id = last_max_frame.id + 1
            sum_end_id = max_diff_id - 1

            id_no = sum_start_id
            sum_diff = 0
            while True:

                sum_frame_item = list_frames[id_no]
                sum_diff += sum_frame_item.diff
                id_no += 1
                if id_no > sum_end_id:
                    break

            average_diff = sum_diff / (sum_end_id - sum_start_id + 1)
            if max_diff_frame.diff >= (m_suddenJudge * average_diff):
                possible_frame.append(max_diff_frame)
                window_frame = []
                index = possible_frame[-1].id + m_MinLengthOfShot
                continue
            else:
                index = max_diff_frame.id + 1
                window_frame = []
                continue
        """
        
        get the index of the first and last frame of a shot
        
        """
        for i in range(0, len(possible_frame)):
            start_id_spot.append(possible_frame[i].id)
            end_id_spot.append(possible_frame[i].id - 1)

        sus_last_frame = possible_frame[-1]
        last_frame = list_frames[-1]
        if sus_last_frame.id < last_frame.id:
            possible_frame.append(last_frame)
            end_id_spot.append(possible_frame[-1].id)

        return possible_frame, start_id_spot, end_id_spot

    def optimize_frame(self, tag_frames, list_frames):
        '''
            optimize the possible frame
        '''
        new_tag_frames = []
        frame_count = 10
        diff_threshold = 10
        diff_optimize = 2
        start_id_spot = []
        start_id_spot.append(0)
        end_id_spot = []

        for tag_frame in tag_frames:

            tag_id = tag_frame.id
            """
            
            check whether the difference of the possible frame is no less than 10.
            
            """
            if tag_frame.diff < diff_threshold:
                continue
            """
            
            check whether the difference is more than twice the average difference of 
            the previous 10 frames and the subsequent 10 frames.
            
            """
            #get the previous 10 frames
            pre_start_id = tag_id - frame_count
            pre_end_id = tag_id - 1
            if pre_start_id < 0:
                continue

            pre_sum_diff = 0
            check_id = pre_start_id
            while True:
                pre_frame_info = list_frames[check_id]
                pre_sum_diff += pre_frame_info.diff
                check_id += 1
                if check_id > pre_end_id:
                    break

            #get the subsequent 10 frames
            back_start_id = tag_id + 1
            back_end_id = tag_id + frame_count
            if back_end_id >= len(list_frames):
                continue

            back_sum_diff = 0
            check_id = back_start_id
            while True:
                back_frame_info = list_frames[check_id]
                back_sum_diff += back_frame_info.diff
                check_id += 1
                if check_id > back_end_id:
                    break

            # calculate the difference of the previous 10 frames and the subsequent 10 frames
            sum_diff = pre_sum_diff + back_sum_diff
            average_diff = sum_diff / (frame_count * 2)

            #check whether the requirement is met or not
            if tag_frame.diff > (diff_optimize * average_diff):
                new_tag_frames.append(tag_frame)
        """
        get the index of the first and last frame of a shot
        """

        for i in range(0, len(new_tag_frames)):
            start_id_spot.append(new_tag_frames[i].id)
            end_id_spot.append(new_tag_frames[i].id - 1)

        last_frame = list_frames[-1]
        if new_tag_frames[-1].id < last_frame.id:
            new_tag_frames.append(last_frame)

        end_id_spot.append(new_tag_frames[-1].id)

        return new_tag_frames, start_id_spot, end_id_spot


def shot_transition_detection(videopath: str, output_path_to_shots: str):
    cap = cv2.VideoCapture(videopath)
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    frames = []
    success, frame = cap.read()
    i = 0
    FRAME = Frame(0, 0)
    while (success):
        luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = luv
        """
        
        calculate the difference between frames 
        
        """

        if curr_frame is not None and prev_frame is not None:
            diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(diff)
            diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)
        elif curr_frame is not None and prev_frame is None:
            diff_sum_mean = 0
            frame_diffs.append(diff_sum_mean)
            frame = Frame(i, diff_sum_mean)
            frames.append(frame)

        prev_frame = curr_frame
        i = i + 1
        success, frame = cap.read()
    cap.release()

    #detect the possible frame
    frame_return, start_id_spot_old, end_id_spot_old = FRAME.find_possible_frame(
        frames)

    #optimize the possible frame
    new_frame, start_id_spot, end_id_spot = FRAME.optimize_frame(
        frame_return, frames)

    #store the result
    start = np.array(start_id_spot)[np.newaxis, :]
    end = np.array(end_id_spot)[np.newaxis, :]
    spot = np.concatenate((start.T, end.T), axis=1)

    pathlib.Path(f'{output_path_to_shots}').mkdir(parents=True, exist_ok=True)

    np.savetxt(f'{output_path_to_shots}/intermediate_result.txt',
               spot,
               fmt='%d',
               delimiter='\t')


def convert_intermediate_shot_results(
    path_to_video: str,
    path_to_intermediate_shot_data: str,
) -> None:
    """Converts heuristic shot detection output to csv file and 
    writes to output path.

    Attributes:
        path_to_video: Path to a video file.
        path_to_intermediate_shot_data: Output Path to a video's 
        shot data directory.
    """
    vidcap = cv2.VideoCapture(path_to_video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    frame_timestamp_list = read_intermediate_shot_results(
        path_to_intermediate_shot_data, fps)
    write_shot_results(path_to_intermediate_shot_data, frame_timestamp_list)


def read_intermediate_shot_results(
    path_to_shot_data: str,
    fps: float,
) -> Sequence[float]:
    """Read shot detection output and create list each shots timestamp.

    Attributes:
        path_to_shot_data: Path to a video's shot data directory.
        fps: Frame per seconds.
        
    Return:
        List of each shots timestamp from start_time to end_time.    
    """
    shot_timestamps = []
    shot_number = 1
    with open(f'{path_to_shot_data}/intermediate_result.txt') as f:
        for start_time, end_time in csv.reader(f, delimiter='\t'):
            frame_start_time = int(start_time) / fps
            frame_end_time = int(end_time) / fps
            shot_timestamps.append(
                [shot_number, frame_start_time, frame_end_time])
            shot_number += 1
    return shot_timestamps


def write_shot_results(
    path_to_shot_data: str,
    shot_timestamps: Sequence[float],
) -> None:
    """Write shot change results to output path.

    Attributes:
        path_to_shot_data: Output path to a video's shot data directory.
        shot_timestamps: List of each shots timestamp from start_time to 
        end_time.
    """
    header = ['shot', 'start_time', 'end_time']
    with open(f'{path_to_shot_data}/shots.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(shot_timestamps)
