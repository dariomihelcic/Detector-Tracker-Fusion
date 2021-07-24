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

SAVE_DATA = 0

BB_method = []
count_frames = 0
count_matches = 0

TIME_THRESHOLD = 0.03

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
    file_name = 'C:/Users/Dario/'+RECORDING_NAME+'_BB_method.csv' 

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
        start = time.time()
        while True:
            if count_frames == 0:
                start = time.time()  
            if count_frames%10 == 0 and count_frames != 0:
                fps = 10/(time.time()-start_10)
            if count_frames%10 == 0 or count_frames == 0:
                start_10 = time.time()

            # Read frame from camera
            ret, image_np = cap.read()

            if image_np is None:
                end = time.time()
                # Check number of matches
                if count_matches!=len(time_stamps):
                    print('WARNING: Not all timestamps were matched! Num: {}\n'.format(count_matches))
                elif SAVE_DATA:
                    print('\nINFO: End of video reached, writing to .csv file.\n')
                    writeToCsv(BB_method)

                duration = (end-start)
                print("Program ran for {} sec.".format(duration))
                print("FPS: {}".format(count_frames/duration))
                cv2.destroyAllWindows()
                break

            (H, W) = image_np.shape[:2]
                    
            timer = cv2.getTickCount()
            image_np = cv2.cvtColor(np.copy(image_np), cv2.COLOR_BGR2RGB)

            # rec_5 was fliped in annotation tool
            
            if NUM_REC == 5:
                image_np = cv2.flip(image_np, 1)
                image_np = cv2.flip(image_np, 0)
            if 1:
                image_np_expanded = np.expand_dims(image_np, axis=0)
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
                class_l = np.squeeze(classes).astype(np.int32)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    box,
                    class_l,
                    score,
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                    )

                #Calculate frame time
                frame_time = count_frames/30
                #print("time stamp current frame:",frame_time)

                try:
                    stamp_time = float(time_stamps[count_matches])
                    difference = abs(frame_time-stamp_time)
                    if difference<TIME_THRESHOLD or frame_time<=0.0:
                        count_matches+=1
                        #Find used boxes
                        for i in range(box.shape[0]):
                            # if timestamp and class match
                            if score[i] > min_score_tresh:
                                # box 0: ymin, 1: xmin, 2: ymax, 3:xmax
                                # denormalize
                                box_det = box[i]
                                box_det[0] *= H
                                box_det[1] *= W
                                box_det[2] *= H
                                box_det[3] *= W  
                                '''        
                                # visually compare
                                cv2.rectangle(image_np,
                                     (box_det[1],box_det[0]),
                                     (box_det[3],box_det[2]),
                                     (50,50,250),2)
                                '''
                                data_row = [time_stamps[count_matches],
                                         box_det[1], box_det[0],
                                         box_det[3], box_det[2]]
                                BB_method.append(data_row)
                                break
                        if not data_row:
                            data_row = [time_stamps[count_matches], 0, 0, 0, 0]
                            BB_method.append(data_row)
                        data_row = []
                except IndexError:
                    print('INFO: End of list reached.')

            count_frames+=1

            #Calculate simulated fps
            fps_sim = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
            cv2.putText(image_np, "FPS : " + str(int(fps_sim)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (250,50,50), 2);

            # Display output
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            cv2.imshow('Detection', cv2.resize(image_np, (W, H)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            if cv2.waitKey(25) & 0xFF == ord('p'):
                cv2.waitKey(0)

