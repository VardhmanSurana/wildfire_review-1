from detection import FireTracker,SmokeTracker
from utils import read_video,save_video

def main():
    #input
    input_video_path = "Input_video/input_video.mp4"
    video_frames = read_video(input_video_path)
    #fire
    fire_tracker = FireTracker(model_path='models/best.pt')
    fire_detection = fire_tracker.detect_frames(video_frames,read_from_stubs = False,stub_path = 'tracker_stubs/fire_detection.pkl')
    output_video_frames = fire_tracker.draw_bboxes(video_frames, fire_detection) 
    

    # smoke
    smoke_tracker = SmokeTracker(model_path='models/smoke.pt')  
    smoke_detection = smoke_tracker.detect_frames(video_frames,read_from_stubs = False,stub_path = 'tracker_stubs/smoke_detection.pkl')
    output_video_frames = smoke_tracker.draw_bboxes(video_frames, smoke_detection) 
    #output
    save_video(output_video_frames, "output_video/output_videos1.avi")

if __name__ == "__main__":
    main()