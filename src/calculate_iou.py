import numpy as np
import csv
import cv2

# number of recording 1-7
NUM_REC = 2 
# which tracker to use 
# 0: detector, 1: KCF, 2: CSRT, 3: fused_MF, 4: MF
PROGRAM = 5

NUM_EXAMPLES = 50

for i in range(7):
	NUM_REC = i+1
	PATH_TO_GROUNDTRUTH = 'C:/Users/Dario/object_tracking/groundtruth/GT_rec_'
	PATH_TO_GROUNDTRUTH += str(NUM_REC) + '.csv'

	if PROGRAM == 0:
		# Only the new detector
		PATH_TO_METHOD_DATA = 'C:/Users/Dario/rec_'+str(NUM_REC)+'_BB_method.csv'
	elif PROGRAM == 1:
		# KCF tracker
		PATH_TO_METHOD_DATA = 'C:/Users/Dario/rec_'+str(NUM_REC)+'_BB_tracker.csv'
	elif PROGRAM == 2:
		# CSRT tracker
		PATH_TO_METHOD_DATA = 'C:/Users/Dario/rec_'+str(NUM_REC)+'_BB_CSRT.csv'
	elif PROGRAM == 3:
		# detector + MF tracker fusion
		PATH_TO_METHOD_DATA = 'C:/Users/Dario/rec_'+str(NUM_REC)+'_BB_fused.csv'
	elif PROGRAM == 4:
		# MedianFlow tracker
		PATH_TO_METHOD_DATA = 'C:/Users/Dario/rec_'+str(NUM_REC)+'_BB_MF.csv'	

	def calculate_iou(a, b):
	    x1 = max(a[0], b[0])
	    y1 = max(a[1], b[1])
	    x2 = min(a[2], b[2])
	    y2 = min(a[3], b[3])

	    width = (x2 - x1)
	    height = (y2 - y1)
	    # handle case where there is NO overlap
	    if (width<0) or (height <0):
	        return 0.0
	    area_overlap = width * height

	    area_a = (a[2] - a[0]) * (a[3] - a[1])
	    area_b = (b[2] - b[0]) * (b[3] - b[1])
	    area_combined = area_a + area_b - area_overlap

	    iou = area_overlap / area_combined
	    return iou

	time_stamps = []
	bb_groundtruth = []
	image_names = []
	with open(PATH_TO_GROUNDTRUTH, 'r') as csvfile:
		csv_gt = csv.reader(csvfile)
		for i, row in enumerate(csv_gt):

			if i > 0:
				image_names.append(row[0])
				time_stamps.append(row[0].split('=')[1]) 
				box = [float(row[1]), float(row[2]),\
					   float(row[3]), float(row[4])]
				bb_groundtruth.append(box)
	time_stamps_m = []
	bb_method = []
	with open(PATH_TO_METHOD_DATA, 'r') as csvfile:
		csv_md = csv.reader(csvfile)
		for i, row in enumerate(csv_md):
			if i > 0:
				time_stamps_m.append(row[0])
				box = [float(row[1]), float(row[2]),\
					   float(row[3]), float(row[4])]
				bb_method.append(box)

	# Sort BB by timestamp
	idx = np.argsort(time_stamps)	
	time_stamps = ([time_stamps[i] for i in idx])
	bb_groundtruth = ([bb_groundtruth[i] for i in idx])	
	image_names = ([image_names[i] for i in idx])	
	#print(time_stamps)
	#print(time_stamps_m)
	result_list = []
	if len(bb_groundtruth)==len(bb_method):

		for i, time in enumerate(time_stamps):
			result = calculate_iou(bb_groundtruth[i], bb_method[i])
			result_list.append(result)
			#print(result*100)
	else: 
		print('WARNING: Number of boxes does not match!')
		print(len(bb_groundtruth))
		print(len(bb_method))

	final_result = 0
	for num in result_list:
		final_result += num
	final_result /= len(time_stamps)

	print("Final result for rec_{} is:{:.2f}%".format(NUM_REC, final_result*100))

# Comapre bb
for i in range(NUM_EXAMPLES):
	#print(time_stamps[1+i])
	#print(time_stamps_m[1+i])
	try:
		image_path = 'object_tracking/data/vott-csv-export/'+image_names[1+i]+'.jpg'
		image = cv2.imread(image_path)

		# red box: groundtruth
		start_point = int(bb_groundtruth[1+i][0]),int(bb_groundtruth[1+i][1])
		end_point = int(bb_groundtruth[1+i][2]),int(bb_groundtruth[1+i][3])
		cv2.rectangle(image, start_point, end_point,(50,50,250),2)

		# blue box: method
		start_point = int(bb_method[1+i][0]),int(bb_method[1+i][1])
		end_point = int(bb_method[1+i][2]),int(bb_method[1+i][3])
		cv2.rectangle(image, start_point, end_point,(250,50,50),2)
		'''
		if(i%2==1):
			im_path = 'C:/Users/Dario/Desktop/img_'+str(i+1)+'.jpg'
			cv2.imwrite(im_path, image)
		'''
		cv2.imshow("Draw BB", image);
		if cv2.waitKey(0) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
	except:
		continue
cv2.destroyAllWindows()

