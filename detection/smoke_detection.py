from ultralytics import YOLO
import pickle
import cv2
import numpy as np

class SmokeTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.frame_counter = 0

    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        smoke_detector = []

        if read_from_stubs and stub_path is not None:
            with open(stub_path, 'rb') as f:
                smoke_detector = pickle.load(f)
            return smoke_detector

        for frame in frames:
            self.frame_counter += 1
            # if self.frame_counter % 12 == 0:  # Detect every 12th frame
            smoke_detect = self.detect_frame(frame)
            smoke_detector.append(smoke_detect)
            # else:
            #     smoke_detector.append({})  # Append empty dict for non-detection frames

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(smoke_detector, f)
        return smoke_detector
    def detect_frame(self,frame):
        result = self.model.predict(frame,conf = 0.16)[0]
        smoke_dict = {}
        for box in result.boxes:
            result = box.xyxy.tolist()[0]
            smoke_dict[1] = result
        return smoke_dict

    def draw_bboxes(self,video_frames,smoke_detections):
        output_video_frames = []
        for frame ,smoke_dict in zip(video_frames,smoke_detections):
            for track_id,bbox in smoke_dict.items():
                x1,y1,x2,y2 = bbox
                #     coordinates x1 = x min,y1= y min & x2 = x max,y2= y max   Color in RGB format , 2 - not filled just border
                cv2.putText(frame, f"Smoke",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            output_video_frames.append(frame)
        return output_video_frames