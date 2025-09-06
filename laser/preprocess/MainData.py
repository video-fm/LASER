import json
import os
import numpy as np
import cv2

class MainData:
    def __init__(self, video_dir, save_path, video_segment_length, frames_per_second):
        self.video_dir = video_dir
        self.save_path = save_path
        self.video_segment_length = video_segment_length
        self.frames_per_second = frames_per_second

    def timestamp_to_frame(self, video_path, timestamp):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        video.release()

        return frame_number

    def load_video(self, video_path, start_time, end_time, target_shape=None):
        if not os.path.exists(video_path):
            print("video path does not exist")
            return []

        cap = cv2.VideoCapture(video_path)
        video = []

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Convert start_time and end_time to frames
        start_frame = self.timestamp_to_frame(video_path, start_time)
        end_frame = self.timestamp_to_frame(video_path, end_time)

        # Calculate the interval between frames (sample every nth frame)
        frame_interval = int(fps // self.frames_per_second)

        current_frame = 0

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                # Only process frames between start_frame and end_frame
                if start_frame <= current_frame < end_frame:
                    if current_frame % frame_interval == 0:
                        if target_shape is not None:
                            frame = cv2.resize(frame, target_shape)
                        video.append(frame)
                elif current_frame >= end_frame:
                    break

                current_frame += 1
            else:
                break

        cap.release()

        if not video:
            print("No frames captured")
            return []

        return np.stack(video)


    def count_seconds(self, video_path):
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps > 0:
            duration_seconds = total_frames / fps
            print(f"Video duration: {duration_seconds} seconds")
            return duration_seconds
        else:
            print("Could not determine frame rate")
            return 0


    def generate_data(self):
        if os.path.exists(self.save_path):
            all_results = json.load(open(self.save_path, 'r'))
        else:
            all_results = {}


        for video_id_mp4 in os.listdir(self.video_dir):
            video_id = video_id_mp4.replace(".mp4", "")
            video_path = os.path.join(self.video_dir, video_id_mp4)
            total_seconds = self.count_seconds(video_path)

            ## Split the video into semgents
            start_time = 0
            end_time_list = [min(i + self.video_segment_length, total_seconds) for i in range(0, int(total_seconds), self.video_segment_length)]


            for i, end_time in enumerate(end_time_list):
                if video_id in all_results:
                    if i in all_results[video_id]:
                        continue
                else:
                    all_results[video_id] = {}


                all_results[video_id][i] = {
                    'start_frame': self.timestamp_to_frame(video_path, start_time),
                    'end_frame': self.timestamp_to_frame(video_path, end_time)
                }

                start_time = end_time

            json.dump(all_results, open(self.save_path, 'w'))