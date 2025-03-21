import json
import glob
import numpy as np
import cv2
import os
import face_recognition
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class FaceExtractor:
    def __init__(self, input_dir, output_dir, min_frames=150, max_frames=150, 
                 batch_size=4, resize_dim=(112, 112), fps=30):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.resize_dim = resize_dim
        self.fps = fps
        
        os.makedirs(output_dir, exist_ok=True)
        
    def get_video_files(self):
        video_files = glob.glob(f"{self.input_dir}/*.mp4")
        qualified_videos = []
        frame_counts = []
        
        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if frame_count >= self.min_frames:
                qualified_videos.append(video_file)
                frame_counts.append(frame_count)
        
        self.video_files = qualified_videos
        self.frame_counts = frame_counts
        
        print(f"Found {len(qualified_videos)} videos with at least {self.min_frames} frames")
        if frame_counts:
            print(f"Average frames per video: {np.mean(frame_counts):.2f}")
        
        return qualified_videos
    
    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx > self.max_frames:
                break
                
            frames.append(frame)
            frame_idx += 1
        
        cap.release()
        return frames
    
    def batch_frames(self, frames):
        for i in range(0, len(frames), self.batch_size):
            yield frames[i:i + self.batch_size]
    
    def process_video(self, video_path):
        output_path = os.path.join(self.output_dir, os.path.basename(video_path))
        
        if os.path.exists(output_path):
            print(f"Skipping existing file: {output_path}")
            return False
        
        try:
            frames = self.extract_frames(video_path)
            
            if not frames:
                print(f"No frames extracted from {video_path}")
                return False
            
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc('M','J','P','G'),
                self.fps,
                self.resize_dim
            )
            
            face_count = 0
            
            for batch in self.batch_frames(frames):
                face_locations = face_recognition.batch_face_locations(batch)
                
                for i, face_list in enumerate(face_locations):
                    if face_list:  # If at least one face was found
                        for face in face_list[:1]:  # Process only the first face
                            top, right, bottom, left = face
                            face_img = batch[i][top:bottom, left:right]
                            
                            if 0 not in face_img.shape:  # Check if face crop is valid
                                resized_face = cv2.resize(face_img, self.resize_dim)
                                out.write(resized_face)
                                face_count += 1
            
            out.release()
            
            if face_count == 0:
                os.remove(output_path)  # Remove empty videos
                print(f"No faces detected in {video_path}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    def process_all_videos(self, num_workers=4):
        if not hasattr(self, 'video_files'):
            self.get_video_files()
            
        already_processed = glob.glob(f"{self.output_dir}/*.mp4")
        print(f"Found {len(already_processed)} already processed videos")
        
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_video, video_path) 
                      for video_path in self.video_files]
            
            for future in tqdm(futures, total=len(futures), desc="Processing videos"):
                if future.result():
                    processed_count += 1
        
        print(f"Successfully processed {processed_count} videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from videos")
    parser.add_argument("--input", type=str, default="/content/Real videos", help="Input directory")
    parser.add_argument("--output", type=str, default="./Face_only_data", help="Output directory")
    parser.add_argument("--min-frames", type=int, default=150, help="Minimum frames required")
    parser.add_argument("--max-frames", type=int, default=150, help="Maximum frames to process")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for face detection")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    
    extractor = FaceExtractor(
        input_dir=args.input,
        output_dir=args.output,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        batch_size=args.batch_size
    )
    
    extractor.process_all_videos(num_workers=args.workers)
