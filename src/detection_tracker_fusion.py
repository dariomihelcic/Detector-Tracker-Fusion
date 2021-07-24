import tensorflow as tf
import numpy as np
import time
import cv2
import csv
import os
from utils import label_map_util
from utils import visualization_utils as vis_util

NUM_REC = 4
PATH_TO_RECORDING = 'C:/Users/Dario/rec_'+str(NUM_REC)+'.mp4'
PATH_TO_GROUNDTRUTH = 'C:/Users/Dario/object_tracking/groundtruth/GT_rec_'+str(NUM_REC)+'.csv'
RECORDING_NAME = PATH_TO_RECORDING.split('/')[-1][:-4]

NUM_CLASSES = 7
PATH_TO_FROZEN_GRAPH = 'C:/Users/Dario/models/research/object_detection/frozen_inference_graph.pb'
PATH_TO_LABEL_MAP = 'C:/Users/Dario/models/research/object_detection/label_map.pbtxt'

SAVE_DATA = 1

BB_tracker = []
count_frames = 0
count_matches = 0
fps = 0

TIME_THRESHOLD = 0.03

NUM_TRACKER = 2
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[NUM_TRACKER]

def create_tracker(index):
    tracker_type = tracker_types[index] 
    print(tracker_type)
    if tracker_type == 'BOOSTING':
        return cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        return cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        return cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        return cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        return cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()

tracker = create_tracker(NUM_TRACKER)
cap = cv2.VideoCapture(PATH_TO_RECORDING)

# Extract timestamps from groundtruth file
time_stamps = []
with open(PATH_TO_GROUNDTRUTH, 'r') as csvfile:
    csv_data = csv.reader(csvfile)
    for i, row in enumerate(csv_data):
        if i > 0:
            time_stamps.append(row[0].split('=')[1]) 
    time_stamps = sorted(time_stamps, key = float)

def writeToCsv(data):
    file_name = 'C:/Users/Dario/'+RECORDING_NAME+'_BB_fused_'+tracker_type+'.csv' 

    with open(file_name, mode='w', newline='') as results_file:
        csv_writer = csv.writer(results_file, delimiter=',')
        csv_writer.writerow(['timestamp','xmin','ymin','xmax','ymax'])

        for row in data:
            csv_writer.writerow(row)

#reads the frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:      
        while True:

            if count_frames == 0:
                start = time.time()
                  
            if count_frames%5 == 0 and count_frames != 0:
                fps = 5/(time.time()-start_10)
                print(fps)
            if count_frames%5 == 0 or count_frames == 0:
                start_10 = time.time()

            # Read frame from camera
            ok, image_np = cap.read()

            if image_np is None:
                end = time.time()
                duration = (end-start)
                print("Program ran for {} sec.".format(duration))
                print("FPS: {}".format(count_frames/(end-start)))
                
                # Check number of matches
                if count_matches!=len(time_stamps):
                    print('WARNING: Not all timestamps were matched! Num: {}\n'.format(count_matches))
                elif SAVE_DATA:
                    print('\nINFO: End of video reached, writing to .csv file.\n')
                    writeToCsv(BB_tracker)

                cv2.destroyAllWindows()
                break

            (H, W) = image_np.shape[:2]   

            # rec_5 was fliped in annotation tool
            if NUM_REC == 5:
                image_np = cv2.flip(image_np, 1)
                image_np = cv2.flip(image_np, 0)

            # run detector every 10th frame
            if count_frames%10 == 0:
                frame = np.copy(image_np)
                frame = cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(frame, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                
                min_score_tresh = .5
                score = np.squeeze(scores)
                box = np.squeeze(boxes)

                #Calculate frame time
                frame_time = count_frames/30

                #Find detected BB
                bbox = ()
                for i in range(box.shape[0]):
                    if score[i] > min_score_tresh:
                        box_det = box[i]
                        box_det[0] *= H
                        box_det[1] *= W
                        box_det[2] *= H
                        box_det[3] *= W
                        bbox = (int(box_det[1]), int(box_det[0]),
                                int(box_det[3]-box_det[1]),
                                int(box_det[2]-box_det[0])) 

                if not len(bbox):
                    ok, bbox = tracker.update(image_np)

                else:
                    #initilize new tracker with detected BB
                    del tracker
                    tracker = create_tracker(NUM_TRACKER)
                    ok = tracker.init(image_np, bbox)
                    ok, bbox = tracker.update(image_np)

            else:
                ok, bbox = tracker.update(image_np)

            for i, val in enumerate(bbox):
                if val < 0:
                    lst = list(bbox)
                    lst[i] = 0
                    bbox = tuple(lst)
            
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(image_np, p1, p2, (255,0,0), 2, 1)
                
            else :
                # Tracking failure
                cv2.putText(image_np, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
            frame_time = count_frames/30

            try:
                stamp_time = float(time_stamps[count_matches])
            except:
                continue
            
            difference = abs(frame_time-stamp_time)
            if difference<0.034:
                data_row = [time_stamps[count_matches],
                         bbox[0], bbox[1],
                         bbox[0]+bbox[2], bbox[1]+bbox[3]]
                BB_tracker.append(data_row)
                count_matches+=1

                # Display tracker type on frame
            cv2.putText(image_np, "Fusion Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

            # Display FPS on frame
            cv2.putText(image_np, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

            # Display result
            cv2.imshow("Tracking", image_np)
            
            count_frames+=1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            if cv2.waitKey(25) & 0xFF == ord('p'):
                cv2.waitKey(0)

