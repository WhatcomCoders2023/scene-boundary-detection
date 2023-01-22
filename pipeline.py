from video_processor import VideoProcessor
from shot_transition_detection import *
from generate_face_embeddings import GenerateFaceEmbeddings
from cluster_face_embeddings import ClusterFaceEmbeddings

path_to_video_file = 'movies/nuclearFamily.mp4'
output_path_to_shot_data = 'data/nuclearFamily/shots'
path_to_shot_data = 'data/nuclearFamily/shots'

clustering_algorithm = "DBSCAN"


def pipeline():
    video_processor = VideoProcessor(path_to_video_file, 1)
    video_processor.video_to_frames()

    shot_transition_detection(path_to_video_file, output_path_to_shot_data)
    convert_intermediate_shot_results(path_to_video_file, path_to_shot_data)

    face_embeddings = GenerateFaceEmbeddings(video_processor.path_to_frame_data)
    face_embeddings.generate_embeddings_for_all_frames()

    cluster_embeddings = ClusterFaceEmbeddings(
        face_embeddings.path_to_embeddings, clustering_algorithm)
    cluster_embeddings.generate_people_ids_for_each_frame()


pipeline()