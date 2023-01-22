"""Classes that handles clustering face embeddings to produce labels of person 
in each frame."""

import torch
import glob
import pathlib
import os
import numpy as np

from typing import Any, Mapping, Sequence, Tuple
from sklearn.cluster import DBSCAN


class ClusterFaceEmbeddings:
    """ClusterFaceEmbeddings uses clustering algorithms to generate labels
    ids for person in video.

    Attributes:
        path_to_embeddings: 
        path_to_people_id_to_frames: Path of people ids for each frame in video. 
        cluster: Clustering algorithm used for face embeddings. 
    """

    def __init__(self, path_to_embeddings: str, clustering_algorithm: str):
        self.path_to_embeddings = path_to_embeddings
        self.path_to_people_id_to_frames = self._get_path_of_people_id_to_frames(
        )
        self.cluster = self._get_clustering_algorithm(clustering_algorithm)

    def _get_path_of_people_id_to_frames(self):
        """Gets path of people ids for each frame in video. 

        Returns:
            Path of people ids for each frame in video. 
        """
        base_path = pathlib.Path(self.path_to_embeddings).parent
        path_to_people_id_to_frames = f'{base_path}/faces/'
        return path_to_people_id_to_frames

    def _get_clustering_algorithm(self, clustering_algorithm: str) -> Any:
        """Selects clustering algorithm used for class.

        Attributes:
            clustering_algorithm: Name of algorithm used for clustering.
        
        Returns:
            Clustering algorithm used on face embeddings.
        """
        if clustering_algorithm == 'DBSCAN':
            cluster = DBSCAN(eps=0.75, metric='euclidean', n_jobs=-1)
        return cluster

    def load_frames_to_embeddings(self) -> Mapping[str, np.ndarray]:
        """Loads all embeddings from video to a map of frame path to embedding list.

        Returns:
            Path to a video's embedding data directory.
        """
        embedding_paths = glob.glob(f'{self.path_to_embeddings}/*')

        frame_to_embeddings = {}
        for embedding_folder in embedding_paths:
            for embedding_path in os.listdir(embedding_folder):
                embedding = torch.load(
                    f'{embedding_folder}/{embedding_path}').detach().numpy()
                if embedding_folder not in frame_to_embeddings:
                    frame_to_embeddings[embedding_folder] = []
                frame_to_embeddings[embedding_folder].append(embedding)
        return frame_to_embeddings

    def get_embeddings_to_frames(
        self,
        frames_to_embeddings: Mapping[str, Sequence[np.ndarray]],
    ) -> Tuple[Mapping[int, str], Sequence[np.ndarray]]:
        """Gets path to embedding data folder from frames data folder.

        Attributes:
            path_to_frames: Path to a video's frame data directory.
        
        Returns:
            Path to a video's embedding data directory.
        """

        embedding_index_to_frame_path = {}
        embedding_index = 0
        embedding_list = []
        for frame_path, embeddings in frames_to_embeddings.items():
            for embedding in embeddings:
                embedding_list.append(embedding)
                embedding_index_to_frame_path[embedding_index] = frame_path
                embedding_index += 1
        return embedding_index_to_frame_path, embedding_list

    def cluster_embeddings(
        self,
        embeddings: Sequence[np.ndarray],
    ) -> Sequence[int]:
        """Cluster face embeddings to produce list of people ids for entire video.

        Attributes:
            embeddings: List of face embeddings.
        
        Returns:
            List of people ids for entire video.
        """
        # reduce to embeddings to 2d for DBSCAN
        embeddings = np.squeeze(embeddings, axis=1)

        self.cluster.fit_predict(embeddings)
        return self.cluster.labels_

    def create_frame_path_to_people_ids_map(
        self,
        embedding_index_to_frame_path: Mapping[int, str],
        labels: Sequence[int],
    ) -> Mapping[str, Sequence[int]]:
        """Creates a map of frame path to people ids.

        Attributes:
            embedding_index_to_frame_path: Map of embedding index to frame path.
            labels: List of people ids (represented as ints).
        
        Returns:
            Map of frame path to list of people ids.
        """

        frame_path_to_people_ids = {}
        for i, (_,
                frame_path) in enumerate(embedding_index_to_frame_path.items()):
            if not frame_path in frame_path_to_people_ids:
                frame_path_to_people_ids[frame_path] = []
            frame_path_to_people_ids[frame_path].append(labels[i])
        return frame_path_to_people_ids

    def write_people_id_to_frames(
        self,
        frame_path_to_people_id: Mapping[str, Sequence[int]],
    ) -> None:
        """Writes people id to each frame in video.

        Attributes:
            frame_path_to_people_id: Directory of frame path to people ids.
        """

        pathlib.Path(self.path_to_people_id_to_frames).mkdir(parents=True,
                                                             exist_ok=True)
        for frame_path, people_ids in frame_path_to_people_id.items():
            frame_number = frame_path.split('/')[-1]
            f = open(f'{self.path_to_people_id_to_frames}/{frame_number}', 'w')
            f.write(f'{people_ids}')

    def generate_people_ids_for_each_frame(self) -> None:
        """Generates people ids for each frame in video.
        """
        frames_to_embeddings = self.load_frames_to_embeddings()
        embeddings_to_frames, embeddings = self.get_embeddings_to_frames(
            frames_to_embeddings)
        labels = self.cluster_embeddings(embeddings)
        frame_path_to_people_ids = self.create_frame_path_to_people_ids_map(
            embeddings_to_frames, labels)
        self.write_people_id_to_frames(frame_path_to_people_ids)
