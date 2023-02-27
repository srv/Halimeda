#!/usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # -1 --> Do not use CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
conda_path="/home/sparus/anaconda3/bin:$PATH"
import sys

import numpy as np
import tensorflow as tf
import copy

path2scripts = '/home/sparus/object_detection/models/research' # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts) # making scripts in models/research available for import
# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import config_util
from object_detection.builders import model_builder

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from mine_detection.msg import det
from mine_detection.msg import mine_detections
from std_msgs.msg       import Float64


class Object_detection:


	def __init__(self, name):
		self.name = name

		self.period = rospy.get_param('mine_detec/period')

		self.shape

		self.model_path_ss = "path/to/model.h5"
		self.od_model_path = "path/to/model.h5"
		

		#example of reading from a yaml: self.model_path = rospy.get_param('mine_detec/model_path')

		# Params
		self.init = False
		self.new_image = False
	

		self.confidence = Float64()

		self.det = det()
		self.mine_detections_out = mine_detections()
		self.mine_detections_list = []
		self.s_list = []

		# Set subscribers
		image_sub_ss = message_filters.Subscriber('/stereo_down/scaled_x2/left/image_rect_color', Image)
		image_sub_od = message_filters.Subscriber('/stereo_down/scaled_x2/left/image_rect_color', Image)
		info_sub = message_filters.Subscriber('/stereo_down/left/camera_info', CameraInfo)

		image_sub_ss.registerCallback(self.cb_image_ss)
		image_sub_od.registerCallback(self.cb_image_od)
		info_sub.registerCallback(self.cb_info)

		# Set publishers
		self.pub_mine_det = rospy.Publisher('mine_det', mine_detections, queue_size=4)

		# Set classification timer
		rospy.Timer(rospy.Duration(self.period), self.run)


	def cb_image_ss(self, image):
		image_ss = image
		header = image_ss.header
		info = self.info
		self.init_ss = False
		self.ready_ss = True

		if self.init_ss == False:
			# ss init
			self.init_ss = True

		if  self.ready_od:
			self.ready_ss = False
			# ss inference


	def cb_image_od(self, image):
		image_od = image
		header = image_od.header
		info = self.info
		self.init_od = False
		self.ready_od = True

		if self.init_od == False:
			# od init
			self.init_od = True

		if self.ready_ss:
			self.ready_od = False
			# od inference


	def cb_info(self, info):
		self.info = info




	def run(self,_):

		try:
			image = self.image
			header = self.image.header
			info = self.info

			info.width = int(info.width/self.decimation)
			info.height = int(info.height/self.decimation)

			self.mine_detections_out.header = header
			self.mine_detections_out.image_rect_color = image
			self.mine_detections_out.camera_info = info

			if not self.init:
				rospy.loginfo('[%s]: Start object detection', self.name)	
		except:
			rospy.logwarn('[%s]: There is no input image to run the detection', self.name)
			return

		# Set model
		if not self.init: 
			self.set_models()
			self.init = True
			print("Model init")

		# Object detection
		image_np = np.array(np.frombuffer(image.data, dtype=np.uint8).reshape(720, 960,3))
		input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
		detections = self.detection(input_tensor)

		# check number of detections
		num_detections = int(detections.pop('num_detections'))
		# filter out detections
		detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
		# detection_classes to ints
		detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
		# defining what we need from the resulting detection dict that we got from model output
		key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
		# filter out detections dict to get only boxes, classes and scores
		detections = {key: value for key, value in detections.items() if key in key_of_interest}
		if self.box_th != 0: # filtering detection if a confidence threshold for boxes was given as a parameter
			for key in key_of_interest:
				scores = detections['detection_scores']
				current_array = detections[key]
				filtered_current_array = current_array[scores > self.box_th]
				detections[key] = filtered_current_array
		
		if self.nms_th != 0: # filtering rectangles if nms threshold was passed in as a parameter
			# creating a zip object that will contain model output info as
			output_info = list(zip(detections['detection_boxes'],
									detections['detection_scores'],
									detections['detection_classes']))
			boxes, scores, classes = self.nms(output_info, self.nms_th)
			
			detections['detection_boxes'] = boxes # format: [y1, x1, y2, x2]
			detections['detection_scores'] = scores
			detections['detection_classes'] = classes

		# Publishers
		self.confidence.data = 0
		self.mine_detections_out.num_detections = 0
		self.mine_detections_out.dets = []

		if len(detections['detection_scores'])>0:
			self.confidence.data = detections['detection_scores'][0]
			self.mine_detections_out.num_detections = len(detections['detection_scores'])
			self.s_list.append(detections['detection_scores'][0])

			for i in range(len(detections['detection_scores'])):
				self.det.y1 = int(detections['detection_boxes'][i][0]*info.width)
				self.det.x1 = int(detections['detection_boxes'][i][1]*info.height)
				self.det.y2 = int(detections['detection_boxes'][i][2]*info.width)
				self.det.x2 = int(detections['detection_boxes'][i][3]*info.height)
				self.det.score = detections['detection_scores'][i]
				self.det.object_class = "mine"
				det2 = copy.deepcopy(self.det)
				self.mine_detections_out.dets.append(det2)
		else:
			self.s_list.append(0)

		self.pub_mine_det.publish(self.mine_detections_out)

		#image_bb = self.msgify(Image, image_np_bb, encoding='rgb8')
		#image_bb.header = header
		#self.pub_mine_bb.publish(image_bb)


	#def msgify(msg_type, numpy_obj, *args, **kwargs):
	#	conv = _from_numpy.get((msg_type, kwargs.pop('plural', False)))
	#	return conv(numpy_obj, *args, **kwargs)


	def detection(self,image):
		"""
		Detect objects in image.

		Args:
		image: (tf.tensor): 4D input image

		Returs:
		detections (dict): predictions that model made
		"""

		image, shapes = self.detection_model.preprocess(image)
		prediction_dict = self.detection_model.predict(image, shapes)
		detections = self.detection_model.postprocess(prediction_dict, shapes)
		return detections


	def nms(self,rects, thd=0.5):
		"""
		Filter rectangles
		rects is array of oblects ([x1,y1,x2,y2], confidence, class)
		thd - intersection threshold (intersection divides min square of rectange)
		"""
		out = []

		remove = [False] * len(rects)

		for i in range(0, len(rects) - 1):
			if remove[i]:
				continue
			inter = [0.0] * len(rects)
			for j in range(i, len(rects)):
				if remove[j]:
					continue
				inter[j] = self.intersection(rects[i][0], rects[j][0]) / min(self.square(rects[i][0]), self.square(rects[j][0]))

			max_prob = 0.0
			max_idx = 0
			for k in range(i, len(rects)):
				if inter[k] >= thd:
					if rects[k][1] > max_prob:
						max_prob = rects[k][1]
						max_idx = k

			for k in range(i, len(rects)):
				if (inter[k] >= thd) & (k != max_idx):
					remove[k] = True

		for k in range(0, len(rects)):
			if not remove[k]:
				out.append(rects[k])

		boxes = [box[0] for box in out]
		scores = [score[1] for score in out]
		classes = [cls[2] for cls in out]
		return boxes, scores, classes


	def intersection(self,rect1, rect2):
		"""
		Calculates square of intersection of two rectangles
		rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
		return: square of intersection
		"""
		x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
		y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
		overlapArea = x_overlap * y_overlap;
		return overlapArea


	def square(self,rect):
		"""
		Calculates square of rectangle
		"""
		return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])


if __name__ == '__main__':
	try:
		rospy.init_node('detect_image')
		Object_detection(rospy.get_name())

		rospy.spin()
	except rospy.ROSInterruptException:
		pass
