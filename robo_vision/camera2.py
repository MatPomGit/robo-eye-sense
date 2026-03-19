"""
NIE USUWAĆ MI TEGO! 
PLIK DO TESTÓW d435i!!
"""
import cv2
from realsense_camera import *
from mask_rcnn import *
# Load Realsense camera and Mask R-CNN
rs = RealsenseCamera()
mrcnn = MaskRCNN()


while True:
    # Get frame in real time from Realsense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()
    print(depth_frame)
    
    cv2.imshow("depth_frame", depth_frame)
    cv2.imshow("Bgr frame", bgr_frame)

    # Get object mask
    boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

    # Draw object mask
    bgr_frame = mrcnn.draw_object_mask(bgr_frame)
    
    # Show depth info of the objects
    mrcnn.draw_object_info(bgr_frame, depth_frame)



