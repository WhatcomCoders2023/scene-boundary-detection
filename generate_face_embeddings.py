"""Classes that handle generation of face embeddings from frames (images)."""

import cv2
import facenet_pytorch
import glob
import PIL
import pathlib
import numpy as np
import torch

from typing import Sequence


class GenerateFaceEmbeddings:
    """GenerateFaceEmbeddings generates face embeddings for a 
    directory of frame files.

    Attributes:
        path_to_frames: Path to video file.
        path_to_embeddings: Frame per seconds. Ex: 10 fps = we get 10 images per 
        second of video
        mtcnn_model: Name of video.
        facenet_model: 
    """

    def __init__(self, path_to_frames: str):
        self.frame_files = self._get_all_frame_files(path_to_frames)
        self.path_to_embeddings = self._get_path_to_embeddings(path_to_frames)
        self.mtcnn_model = facenet_pytorch.MTCNN(keep_all=True)
        self.facenet_model = facenet_pytorch.InceptionResnetV1(
            pretrained='vggface2').eval()

    def _get_all_frame_files(self, path_to_frames: str) -> Sequence[str]:
        """Gets all frame files from frame directory.

        Attributes:
            path_to_frames: Path to a video's frame data directory.
        
        Return:
            List of string path to individual frames of a video.
        """
        return sorted(glob.glob(f'{path_to_frames}/*', recursive=True))

    def _get_path_to_embeddings(self, path_to_frames: str) -> str:
        """Gets path to embedding data folder from frames data folder.

        Attributes:
            path_to_frames: Path to a video's frame data directory.
        
        Returns:
            Path to a video's embedding data directory.
        """
        base_path = pathlib.Path(path_to_frames).parent
        path_to_embeddings = f'{base_path}/embeddings/'
        return path_to_embeddings

    def crop_faces(
            self,
            frame_file: str,
            required_size: tuple(int, int) = (160, 160),
    ) -> Sequence[torch.Tensor]:
        """Crops all faces from a frame.

        Attributes:
            frame_file: Name of frame.
            required_size: Tuple of dimensions for images.
                Default = (160, 160) which is the size of images that
                the facenet model was trained with.
        
        Returns:
            A list of torch tensors which correspond to n detected faces 
            as a (n x 3 x image_size x image_size) tensor
        """
        img = PIL.Image.open(frame_file)
        if np.shape(img) != (160, 160):
            img.resize(required_size)
        faces = self.mtcnn_model(img)
        return faces

    def get_embedding(self,
                      faces: Sequence[torch.Tensor]) -> Sequence[torch.tensor]:
        """Gets embeddings from all faces.
        
        Attributes:
            faces: List of torch tensors which correspond to n detected faces 
            as a (n x 3 x image_size x image_size) tensor
        
        Returns:
            A list of torch tensors which correspond to face embeddings.
        """
        embeddings = []
        for face in faces:
            img_embedding = self.facenet_model(face.unsqueeze(0))
            embeddings.append(img_embedding)
        return embeddings

    def write_embeddings(self, embeddings: Sequence[torch.tensor],
                         path_to_embedding: str) -> None:
        """Write embeddings to video's embedding directory.

        Attributes:
            embeddings: List of torch tensors which correspond to
            face embeddings.
            path_to_embedding: Path to video's embedding directory.
        """
        for i, embedding in enumerate(embeddings):
            embedding_name = f'{path_to_embedding}/embedding_{i+1}.pt'
            torch.save(embedding, embedding_name)

    def generate_embeddings_for_all_frames(self) -> None:
        """Generates embeddings for all frames.

        1) All faces are detected and cropped from an image 
        using MTCNN model.
        2) If faces were detected, FaceNet models generates
        face embeddings for each face
        3) Embeddings are written to video's embedding
        directory.
        """
        for path_to_frame_file in self.frame_files:
            faces = self.crop_faces(path_to_frame_file)
            frame_file_name = pathlib.Path(path_to_frame_file).name.split(
                ".")[0]
            path_to_embedding = f'{self.path_to_embeddings}/{frame_file_name}/'
            pathlib.Path(path_to_embedding).mkdir(parents=True, exist_ok=True)
            if faces != None:
                embeddings = self.get_embedding(faces)
                self.write_embeddings(embeddings, path_to_embedding)
