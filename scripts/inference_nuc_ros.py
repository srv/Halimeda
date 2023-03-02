#!/usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # -1 --> Do not use CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
conda_path="/home/sparus/anaconda3/bin:$PATH"
import sys

import numpy as np
import tensorflow as tf
import torch

import copy

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo

# TODO CHECK THREADING
# https://stackoverflow.com/questions/65678158/in-python-3-how-can-i-run-two-functions-at-the-same-time
# https://nitratine.net/blog/post/python-threading-basics/
# https://stackoverflow.com/questions/3221655/why-doesnt-a-string-in-parentheses-make-a-tuple-with-just-that-string
from threading import Thread, Lock

# import cv2 # TODO NEEDED? CHECK WITH ROS_PATH AND CONDA_PATH
from cv_bridge import CvBridge, CvBridgeError


class Halimeda_detection:


	def __init__(self, name):
		self.name = name

		self.period = rospy.get_param('mine_detec/period')

		self.shape = 1024

		self.model_path_ss = "path/to/model.h5"
		self.model_path_od = "path/to/model.h5"

		self.thr_ss : 82
		self.thr_od : 32

		# CvBridge for image conversion
		self.bridge = CvBridge()
		
		# TODO READ FROM YAML?
		#example of reading from a yaml: self.model_path = rospy.get_param('mine_detec/model_path')

		# Params
		self.init = False
		self.new_image = False
	
		# Set subscribers
		image_sub = message_filters.Subscriber('/stereo_down/left/image_raw', Image)
		info_sub = message_filters.Subscriber('/stereo_down/left/camera_info', CameraInfo)

		image_sub.registerCallback(self.cb_image)
		info_sub.registerCallback(self.cb_info)

		# Set publishers
		self.pub_img_merged = rospy.Publisher('segmented', Image, queue_size=4)

		# Set classification timer
		rospy.Timer(rospy.Duration(self.period), self.run)


	def cb_image(self, image):
		self.image = image
		self.new_image = True


	def cb_info(self, info):
		self.info = info


	def set_models(self):

		self.model_ss = tf.keras.models.load_model(os.path.join(self.model_path_ss, "model.h5"))

		self.model_od = torch.hub.load(self.model_path_od, 'custom', path='weights/best.pt', source='local',force_reload = False)
		# TODO CHECK
		# self.model_od = torch.hub.load(path, 'resnet50', weights='ResNet50_Weights.DEFAULT')
		# https://pytorch.org/docs/stable/hub.html
		self.model_od.to(torch.device('cpu')).eval()


	def run(self,_):
		
		# New image available
		if not self.new_image:
			return
		self.new_image = False

		try:
			image = self.image
			header = self.image.header
			info = self.info

			info.width = self.shape
			info.height = self.shape

			self.pub_img_merged.header = header
			
		except:
			rospy.logwarn('[%s]: There is no input image to run the inference', self.name)
			return

		# Set model
		if not self.init: 
			self.set_models()
			self.init = True
			print("Model init")
			
		rospy.loginfo('[%s]: Starting inferences', self.name)	

		# Object detection
		self.image_np = np.array(np.frombuffer(image.data, dtype=np.uint8).reshape(self.shape, self.shape,3))
		#self.image_np=img_as_ubyte(self.image_np)	# TODO NEEDED?

		thread_ss = Thread(target=self.inference_ss)
		thread_od = Thread(target=self.inference_od)
		thread_ss.start()
		thread_od.start()
		thread_ss.join() # Don't exit while threads are running
		thread_od.join() # Don't exit while threads are running		
		
		image_merged_np = merge(self)
		image_merged = img = self.bridge.cv2_to_imgmsg(image_merged_np, encoding="bgr8")
		image_merged.header = header
		self.pub_img_merged.publish(image_merged)

		
	def inference_ss(self):
		X_test = np.zeros((1, self.shape, self.shape, 3), dtype=np.uint8)
		X_test[0] = self.image_np
		preds_test = self.model_ss.predict(X_test, verbose=1)
		self.image_np_ss = np.squeeze(preds_test[0])
		rospy.loginfo('[%s]: SS inference done', self.name)	
	
	
	def inference_od(self):
		dets_od = self.model_od([self.image_np])

		self.image_np_od = np.zeros([self.shape, self.shape], dtype=np.uint8) 

		dets_pandas = dets_od.pandas().xyxy[0]
		for index, row in dets_pandas.iterrows():
			conf = row['confidence']
			xmin=row['xmin']
			ymin=row['ymin']
			xmax=row['xmax']
			ymax=row['ymax']

			for j in range(ymin, ymax):			# TODO FIX XYXY XYWH ABS REL
				for k in range(xmin, xmax):		# TODO FIX XYXY XYWH ABS REL
					self.image_np_od[j, k] = int(255*conf)

		rospy.loginfo('[%s]: OD inference done', self.name)	


	def merge(self):
		# TODO APPLY BEST MERGING %
		image_merged = self.image_np_ss*0.5 + self.image_np_op*0.5
		return image_merged
		
	# TODO NEEDED?
	# def array2img(self, array):
	# 	img_0 = np.ndarray(shape=(array.shape[0], array.shape[1], 3), dtype=np.uint8)
	# 	img_0[:,:,0] = array*255
	# 	img_0[:,:,1] = array*255
	# 	img_0[:,:,2] = array*255
	# 	img = self.bridge.cv2_to_imgmsg(img_0, encoding="bgr8")
	# 	return img	
		

if __name__ == '__main__':
	try:
		rospy.init_node('detect_halimeda')
		Halimeda_detection(rospy.get_name())

		rospy.spin()
	except rospy.ROSInterruptException:
		pass
