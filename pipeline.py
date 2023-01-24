from video_processor import VideoProcessor
from shot_transition_detection import *
from generate_face_embeddings import GenerateFaceEmbeddings
from cluster_face_embeddings import ClusterFaceEmbeddings
from scene_boundary_detection import SceneBoundaryDetection

movie_name = 'SuperHero'
path_to_video_file = f'movies/{movie_name}.mp4'
output_path_to_shot_data = f'data/{movie_name}/shots'
path_to_shot_csv_data = f'data/{movie_name}/shots/shots.csv'
path_to_people_ids = f'data/{movie_name}/faces'

clustering_algorithm = "DBSCAN"


def pipeline():
    print("Converting video to frames\n")
    video_processor = VideoProcessor(path_to_video_file, 1)
    video_processor.video_to_frames()

    print("Performing Shot Change Detection\n")
    shot_transition_detection(path_to_video_file, output_path_to_shot_data)
    convert_intermediate_shot_results(path_to_video_file,
                                      output_path_to_shot_data)

    print("Generating Face Embeddings\n")
    face_embeddings = GenerateFaceEmbeddings(video_processor.path_to_frame_data)
    face_embeddings.generate_embeddings_for_all_frames()

    print("Clustering Embeddings and assigning people ids to each frames\n")
    cluster_embeddings = ClusterFaceEmbeddings(
        face_embeddings.path_to_embeddings, clustering_algorithm)
    cluster_embeddings.generate_people_ids_for_each_frame()

    shots = ShotsToFrames(
        path_to_shot_csv_data,
        video_processor.frame_number_to_timestamp_milliseconds)
    shots_to_timestamp = shots.shots_to_frames
    shots_to_frame = shots.shots_to_frames

    print("Performing Scene Boundary Detection\n")
    scene_boundary_detection = SceneBoundaryDetection(shots_to_timestamp,
                                                      shots_to_frame,
                                                      path_to_people_ids)
    scene_boundary_detection.run()


pipeline()