import cv2
import csv
import sys
import time

NUM_REC = 5
PATH_TO_RECORDING = 'C:/Users/Dario/rec_'+str(NUM_REC)+'.mp4'
PATH_TO_GROUNDTRUTH = 'C:/Users/Dario/object_tracking/data/GT_org/GT_rec_'+str(NUM_REC)+'.csv'
RECORDING_NAME = PATH_TO_RECORDING.split('/')[-1][:-4]

NUM_TRACKER = 4
SAVE_DATA = 1

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# Extract timestamps from groundtruth file
time_stamps = []
with open(PATH_TO_GROUNDTRUTH, 'r') as csvfile:
    csv_data = csv.reader(csvfile)
    for i, row in enumerate(csv_data):
        if i > 0:
            time_stamps.append(row[0].split('=')[1]) 
    time_stamps = sorted(time_stamps, key = float)

def writeToCsv(data):
    
    if NUM_TRACKER == 2: 
        file_name = 'C:/Users/Dario/'+RECORDING_NAME+'_BB_tracker.csv'
    if NUM_TRACKER == 4: 
        file_name = 'C:/Users/Dario/'+RECORDING_NAME+'_BB_MF.csv'
    if NUM_TRACKER == 7: 
        file_name = 'C:/Users/Dario/'+RECORDING_NAME+'_BB_CSRT.csv'  

    with open(file_name, mode='w', newline='') as results_file:
        csv_writer = csv.writer(results_file, delimiter=',')
        csv_writer.writerow(['timestamp','xmin','ymin','xmax','ymax'])

        for row in data:
            csv_writer.writerow(row)

if __name__ == '__main__' :

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[NUM_TRACKER]

    if int(minor_ver) < 0:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture(PATH_TO_RECORDING)

    # Exit if video not opened.
    if not video.isOpened():
        print('Could not open video')
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    if NUM_REC == 5:
        frame = cv2.flip(frame, 1)
        frame = cv2.flip(frame, 0)

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    count_stamped_frames = 1
    count_matches = 1
    count_frames = 1
    fps = 0
    BB_tracker = []
    start = time.time()
    start_10 = time.time()
    while True:
            
        if count_frames%10 == 0 and count_frames != 0:
            fps = 10/(time.time()-start_10)
        if count_frames%10 == 0 or count_frames == 0:
            start_10 = time.time()

        # Read a new frame
        ok, frame = video.read()

        if NUM_REC == 5:
            frame = cv2.flip(frame, 1)
            frame = cv2.flip(frame, 0)
        if not ok:
            break
        
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        fps_real = video.get(cv2.CAP_PROP_FPS)
        frame_time = count_frames/30
        try:
            stamp_time = float(time_stamps[count_stamped_frames])
        except:
            break

        difference = abs(frame_time-stamp_time)
        if difference<0.03:
            count_matches+=1
            data_row = [time_stamps[count_stamped_frames],
                     bbox[0], bbox[1],
                     bbox[0]+bbox[2], bbox[1]+bbox[3]]
            BB_tracker.append(data_row)
            count_stamped_frames+=1

        count_frames+=1

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : 
            cv2.destroyAllWindows()
            break
    end = time.time()
    if SAVE_DATA:       
        writeToCsv(BB_tracker)
    print("Program ran for {} sec.".format(end-start))
    print("FPS: {}".format(count_frames/(end-start)))
