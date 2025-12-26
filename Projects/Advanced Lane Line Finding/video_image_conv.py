import moviepy
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np

class images_to_video():
    def __init__(self, images, video_path, fps):
        self.image_list = images
        self.video_path = video_path
        self.fps = fps


    def __call__(self):
        if self.image_list.dtype != np.uint8:
            if self.image_list.max() <= 1.0:
                self.image_list = (self.image_list * 255).astype(np.uint8)
            else:
                self.image_list = self.image_list.astype(np.uint8)
        clip = ImageSequenceClip(list(self.image_list), fps=self.fps)
        clip.write_videofile(self.video_path, codec="libx264", audio=False)
        print("Writing Video Completed")



class video_to_images():
    def __init__(self, video_path, fps=None):
        self.video_path = video_path
        self.fps = fps

        
    def __call__(self):    
        clip = VideoFileClip(self.video_path)
        if self.fps:
            clip = clip.set_fps(self.fps)
        frames = clip.iter_frames()
        frames_list = list(frames)
        frames_array = np.array(frames_list)
        return frames_array