import numpy as np
from argparse import ArgumentParser
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
import math
import cv2
from os import walk
import os

np.set_printoptions(suppress=True)

def butter_lowpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y

def linear_interpolation(data):
	_min = data[0]
	diff = (data[-1]-_min)/(len(data)-1)
	y = np.zeros(len(data))
	for i in range(len(data)):
		y[i] = _min + diff*i
	return y

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
	assert(isRotationMatrix(R))

	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

	singular = sy < 1e-6
	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0

	return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta) :
	R_x = np.array([[1, 0, 					0					],
					[0, math.cos(theta[0]),	-math.sin(theta[0])	],
					[0, math.sin(theta[0]),	math.cos(theta[0]) 	]
					])

	R_y = np.array([[math.cos(theta[1]),	0,	math.sin(theta[1])	],
					[0,						1,	0					],
					[-math.sin(theta[1]),	0,	math.cos(theta[1])	]
					])

	R_z = np.array([[math.cos(theta[2]),	-math.sin(theta[2]),	0],
					[math.sin(theta[2]),	math.cos(theta[2]),		0],
					[0,						0,						1]
					])

	R = np.dot(R_z, np.dot( R_y, R_x ))
	return R

def window_filter(euler, window_size):
	datalen = len(euler)
	half_WS = int(window_size/2)
	result = []

	for i in range(datalen):
		if i>=half_WS and i<(datalen-half_WS):
			result.append(np.mean(np.array(euler[(i-half_WS):(i+half_WS+1)]), axis = 0))
		elif i>=half_WS:
			result.append(np.mean(np.array(euler[(i-half_WS):datalen]), axis = 0))
		else:
			result.append(np.mean(np.array(euler[0:(i+half_WS+1)]), axis = 0))

	return result

def euler_linearized(data):
	result = []

	data = np.array(data)
	_x = linear_interpolation(data[:,0])
	_y = linear_interpolation(data[:,1])
	_z = linear_interpolation(data[:,2])
	for i in range(len(_x)):
		result.append([_x[i], _y[i], _z[i]])

	return np.array(result)

def main(args):
	f_camera = open(args.camera_data, "r")

	Camera_3D_data = []
	camera_rotation_data = []
	camera_t_data = []
	camera_f_data = []
	while True:
		line = f_camera.readline()
		if not line:
			break
		if line.find('#timeindex')!=-1:
			cameraData = f_camera.readline().split()
			Camera_3D_data.append([float(cameraData[0]), float(cameraData[1]), float(cameraData[2])])
			t = np.zeros(shape=(3,4)) 
			t[0, 0] = 1
			t[1, 1] = 1
			t[2, 2] = 1
			t[0, 3] = -float(cameraData[0])
			t[1, 3] = -float(cameraData[1])
			t[2, 3] = -float(cameraData[2])

			R = np.zeros(shape=(3,3)) #include internal parameter of camera
			R[2, 0] = float(cameraData[3])
			R[2, 1] = float(cameraData[4])
			R[2, 2] = float(cameraData[5])
			R[0, 0] = float(cameraData[22])
			R[0, 1] = float(cameraData[23])
			R[0, 2] = float(cameraData[24])
			R[1, 0] = float(cameraData[25])
			R[1, 1] = float(cameraData[26])
			R[1, 2] = float(cameraData[27])

			f = np.zeros(shape=(3,3))
			f[0, 0] = float(cameraData[20])/float(cameraData[14])
			f[0, 2] = float(cameraData[18])
			f[1, 1] = float(cameraData[20])/float(cameraData[15])
			f[1, 2] = float(cameraData[19])
			f[2, 2] = 1


			camera_rotation_data.append(R)
			camera_t_data.append(t)
			camera_f_data.append(f)

	Camera_3D_data = np.array(Camera_3D_data)
	print('FrameNum: {}'.format(len(Camera_3D_data)))

	order = 6
	fs = 30.0	# sample rate, Hz
	cutoff = 2.5


	_x = butter_lowpass_filter(Camera_3D_data[:,0], cutoff, fs, order)
	_y = butter_lowpass_filter(Camera_3D_data[:,1], cutoff, fs, order)
	_z = butter_lowpass_filter(Camera_3D_data[:,2], cutoff, fs, order)

	'''
	_x = linear_interpolation(Camera_3D_data[:,0])
	_y = linear_interpolation(Camera_3D_data[:,1])
	_z = linear_interpolation(Camera_3D_data[:,2])
	'''

	smooth_camer_t_data = []
	for i in range(len(_x)):
		t = np.zeros(shape=(3,4)) 
		t[0, 0] = 1
		t[1, 1] = 1
		t[2, 2] = 1
		t[0, 3] = -float(_x[i])
		t[1, 3] = -float(_y[i])
		t[2, 3] = -float(_z[i])
		smooth_camer_t_data.append(t)
	'''
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(_x, _y, _z)
	ax.plot(Camera_3D_data[:,0],Camera_3D_data[:,1],Camera_3D_data[:,2])
	plt.show()
	'''

	camera_eular_data = []
	smooth_camera_rotation_data = []
	for i in range(len(camera_rotation_data)):
		euler = rotationMatrixToEulerAngles(camera_rotation_data[i])
		camera_eular_data.append(euler)

	#smooth_camera_eular_data = window_filter(camera_eular_data, 12)
	smooth_camera_eular_data = euler_linearized(camera_eular_data)

	for i in range(len(smooth_camera_eular_data)):
		_R = np.array(eulerAnglesToRotationMatrix(smooth_camera_eular_data[i]))
		smooth_camera_rotation_data.append(_R)

	smooth_camera_rotation_data = np.array(smooth_camera_rotation_data)

	feature_file_name = []
	for (_, _, filenames) in walk(args.feature_dir):
		feature_file_name.extend(filenames)
	feature_file_name.sort()

	all_feature_GT = []
	all_feature_3D = []
	for filename in feature_file_name:

		fn = os.path.join(args.feature_dir, filename)
		f_pnt = open(fn, 'r')

		feature_GT = []
		feature_3D = []
		while True:
			line = f_pnt.readline()
			if not line:
				break
			featureData = line.split()
			if len(featureData)>2:
				if featureData[3]!='0':
					feature_GT.append([float(featureData[0]), float(featureData[1])])
					feature_3D.append([float(featureData[4]), float(featureData[5]), float(featureData[6])])
		all_feature_GT.append(feature_GT)
		all_feature_3D.append(feature_3D)

	'''
	img = cv2.imread('../img/0007.jpg',1)
	img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_CUBIC)

	predict_coord = []
	for i in range(len(feature_3D)):
		homocoord = np.append(np.array(all_feature_3D[6][i]), [1])
		predict = np.dot(camera_f_data[6], np.dot(smooth_camera_rotation_data[6], np.dot(camera_t_data[6], homocoord))) 
		predict_coord.append([int(predict[0]/predict[2]+0.5*(640-1)), int(predict[1]/predict[2]+0.5*(360-1))])

	for i in range(len(predict_coord)):
		cv2.circle(img,(predict_coord[i][0],predict_coord[i][1]), 2, (0,0,255), -1)
		cv2.circle(img,(int(all_feature_GT[6][i][0]),int(all_feature_GT[6][i][1])), 2, (0,255,0), -1)

	cv2.imwrite('../mretarget_0007.jpg', img)
	'''

	for frameIter in range(len(all_feature_GT)):
		predict_coord = []
		for i in range(len(all_feature_3D[frameIter])):
			homocoord = np.append(np.array(all_feature_3D[frameIter][i]), [1])
			predict = np.dot(camera_f_data[frameIter], np.dot(smooth_camera_rotation_data[frameIter], np.dot(smooth_camer_t_data[frameIter], homocoord))) 
			predict_coord.append([int(predict[0]/predict[2]+0.5*(1280-1)), int(predict[1]/predict[2]+0.5*(720-1))])

		fn = os.path.join(args.warping_coord_dir, '{0:04}.txt'.format(frameIter+1))
		with open(fn, 'w') as feature_f:
			#print("{} {} {} {}".format(frameIter, len(all_feature_3D[frameIter]), len(all_feature_GT[frameIter]), len(predict_coord)))
			for i in range(len(all_feature_3D[frameIter])):
				feature_f.write("{} {} {} 1 {} {}\n".format(i,
					int(all_feature_GT[frameIter][i][1]),
					int(all_feature_GT[frameIter][i][0]),
					predict_coord[i][1],
					predict_coord[i][0]))

	with open('../blender/blender.txt', 'w') as blender_f:
		blender_f.write('#Camera Parameters\n')
		for frameIter in range(len(all_feature_GT)):
			blender_f.write('scene.frame_current = {}\n'.format(frameIter+1))
			blender_f.write('vcam.data.lens = 48.796461\n')
			blender_f.write('vcam.matrix_world = (([{:.6f},{:.6f},{:.6f},0.000000], [{:.6f},{:.6f},{:.6f},0.000000], [{:.6f},{:.6f},{:.6f},0.000000], [{:.6f},{:.6f},{:.6f},1.000000])) \n'.format(
												smooth_camera_rotation_data[frameIter][0,0],
												smooth_camera_rotation_data[frameIter][0,1],
												smooth_camera_rotation_data[frameIter][0,2],
												-smooth_camera_rotation_data[frameIter][1,0],
												-smooth_camera_rotation_data[frameIter][1,1],
												-smooth_camera_rotation_data[frameIter][1,2],
												-smooth_camera_rotation_data[frameIter][2,0],
												-smooth_camera_rotation_data[frameIter][2,1],
												-smooth_camera_rotation_data[frameIter][2,2],
												_x[frameIter], _y[frameIter], _z[frameIter]))
			blender_f.write('vcam.keyframe_insert(\'location\')\n')
			blender_f.write('vcam.keyframe_insert(\'scale\')\n')
			blender_f.write('vcam.keyframe_insert(\'rotation_euler\')\n')
			blender_f.write('vcam.data.keyframe_insert(\'lens\')\n')
			blender_f.write('\n')

			


def parse():
	parser = ArgumentParser()
	parser.add_argument('--camera_data', required=True, help='voodoo camera data')
	parser.add_argument('--feature_dir', required=True, help='voodoo feature directory')
	parser.add_argument('--warping_coord_dir', required=True, help='Filtered feature coordinate')
	parser.add_argument('--img', required=False, help='test img')
	return parser.parse_args()

if __name__ == "__main__":
	main(parse())