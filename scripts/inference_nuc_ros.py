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

# https://stackoverflow.com/questions/65678158/in-python-3-how-can-i-run-two-functions-at-the-same-time
# https://nitratine.net/blog/post/python-threading-basics/
# https://stackoverflow.com/questions/3221655/why-doesnt-a-string-in-parentheses-make-a-tuple-with-just-that-string
from threading import Thread, Lock


#import cv2
#from cv_bridge import CvBridge, CvBridgeError


class Object_detection:


	def __init__(self, name):
		self.name = name

		self.period = rospy.get_param('mine_detec/period')

		self.shape

		self.model_path_ss = "path/to/model.h5"
		self.od_model_path = "path/to/model.h5"
		
		# CvBridge for image conversion
		# self.bridge = CvBridge()
		
		#example of reading from a yaml: self.model_path = rospy.get_param('mine_detec/model_path')

		# Params
		self.init = False
		self.new_image = False
	
		self.det = det()
		self.mine_detections_out = mine_detections()
		self.mine_detections_list = []
		self.s_list = []

		# Set subscribers
		image_sub = message_filters.Subscriber('/stereo_down/scaled_x2/left/image_rect_color', Image)
		info_sub = message_filters.Subscriber('/stereo_down/left/camera_info', CameraInfo)

		image_sub_ss.registerCallback(self.cb_image_ss)
		image_sub_od.registerCallback(self.cb_image_od)
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
		a = 1
		# set ss and od models


	def run(self,_):
		
		# New image available
		if not self.new_image:
			return
		self.new_image = False

		try:
			image = self.image
			header = self.image.header
			info = self.info

			info.width = 1024
			info.height =1024

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
		self.image_np = np.array(np.frombuffer(image.data, dtype=np.uint8).reshape(1024, 1024,3))
		
		thread_ss = Thread(target=self.inference_ss)
		thread_od = Thread(target=self.inference_od)
		thread_ss.start()
		thread_od.start()
		thread_ss.join() # Don't exit while threads are running
		thread_od.join() # Don't exit while threads are running		
		
		image_merged_np = merge(self)
		#image_merged = self.array2img(image_merged_np)
		#image_merged.header = header
		#self.pub_img_merged.publish(image_merged)

		
	def inference_ss(self):
		self.image_np_ss = self.image_np
	
	
	def inference_od(self):
		self.image_np_op = self.image_np

		
	def merge(self):
		image_merged = self.image_np_ss*0.5 + self.image_np_op*0.5
		return image_merged
		
		
	#def array2img(self, array):
	#	img_0 = np.ndarray(shape=(array.shape[0], array.shape[1], 3), dtype=np.uint8)
	#	img_0[:,:,0] = array*255
	#	img_0[:,:,1] = array*255
	#	img_0[:,:,2] = array*255
	#	img = self.bridge.cv2_to_imgmsg(img_0, encoding="bgr8")
	#	return img	
		
if __name__ == '__main__':
	try:
		rospy.init_node('detect_image')
		Object_detection(rospy.get_name())

		rospy.spin()
	except rospy.ROSInterruptException:
		pass
