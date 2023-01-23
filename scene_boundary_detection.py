import ast
import pathlib

from typing import Mapping, MutableSet, Sequence


class SceneBoundaryDetection:
    """SceneBoundaryDetection merges sequence of shots into scenes.

    Attributes:
        shots_to_timestamp: Map of shot number to start and end timestamp.
        shots_to_frames: Map of shot number to list of frames.
        path_to_people_id: Path to people id in video's frame.
    """

    def __init__(self, shots_to_timestamp, shots_to_frames, path_to_people_id):
        self.shots_to_timestamp = shots_to_timestamp
        self.shots_to_frames = shots_to_frames
        self.path_to_people_id = path_to_people_id

    def get_all_people_in_sequence_of_frames(
        self,
        start_frame: float,
        end_frame: float,
    ) -> MutableSet[int]:
        """Gets all people in between two frames.

        Attributes:
            start_frame: Starting frame number.
            end_frame: Ending frame number.

        Return:
            Set of people which represents unique people in
            between two frame indexes.
        """
        people_in_frames = set()
        for frame_number in range(start_frame, end_frame):
            path_to_people_id = f'{self.path_to_people_id}/frame_{frame_number}'
            if pathlib.Path(path_to_people_id).is_file():
                f = open(path_to_people_id, 'r')
                people_ids = ast.literal_eval(f.read())
                for people_id in people_ids:
                    if people_id != -1:
                        people_in_frames.add(people_id)
        return people_in_frames

    def run(self) -> Mapping[int, Sequence[int]]:
        """Run scene boundary detection algorithm.

        1) Iterate through all shots
        2) Merge two shot together if the overlap is large enough.
        3) Else, current collection of shots constitute a distinct scene.

        Return:
            Map of scene number to list of shots.
        """
        scenes_to_shots, scene_number, scene = {}, 1, [1]
        people_in_prev_scene = set()
        for shot_number, frame_list in self.shots_to_frames.items():
            if not frame_list:
                scene.append(shot_number)
                continue
            people_in_current_scene = self.get_all_people_in_sequence_of_frames(
                frame_list[0], frame_list[-1])

            union = len(people_in_current_scene.union(people_in_prev_scene))
            if union != 0:
                overlap = len(
                    people_in_current_scene.intersection(
                        people_in_prev_scene)) / union
            else:
                overlap = 1

            if overlap < 0.05:
                scenes_to_shots[scene_number] = scene
                scene = [shot_number]
                scene_number += 1
            else:
                scene.append(shot_number)

            people_in_prev_scene = people_in_current_scene

        if scene:
            scenes_to_shots[scene_number] = scene
        return scenes_to_shots
