from ultralytics import YOLO
import pickle
import cv2
import numpy as np

class FireTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.frame_counter = 0

    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        fire_detector = []

        if read_from_stubs and stub_path is not None:
            with open(stub_path, 'rb') as f:
                fire_detector = pickle.load(f)
            return fire_detector

        for frame in frames:
            self.frame_counter += 1
            # if self.frame_counter % 12 == 0:  # Detect every 12th frame
            fire_detect = self.detect_frame(frame)
            fire_detector.append(fire_detect)
            # else:
            #     fire_detector.append({})  # Append empty dict for non-detection frames

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(fire_detector, f)

        return fire_detector
    def detect_frame(self,frame):
        result = self.model.predict(frame,conf = 0.16)[0]
        fire_dict = {}
        for box in result.boxes:
            result = box.xyxy.tolist()[0]
            fire_dict[1] = result
        return fire_dict

    def draw_bboxes(self,video_frames, fire_detections):
        output_video_frames = []
        for frame , fire_dict in zip(video_frames, fire_detections):
            for track_id,bbox in  fire_dict.items():
                x1,y1,x2,y2 = bbox
                #     coordinates x1 = x min,y1= y min & x2 = x max,y2= y max   Color in RGB format , 2 - not filled just border
                cv2.putText(frame, f"Fire",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            output_video_frames.append(frame)
        return output_video_frames