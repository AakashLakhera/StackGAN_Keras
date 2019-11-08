import pickle
import numpy as np
import pandas as pd
from PIL import Image

def extract_aux_info(path):
	'''
	Input: file path for embeddings folder
	Output: embeddings, class_info, filenames in that order
	'''
	f1 = open(path + 'filenames.pickle', 'rb')
	filenames = pickle.load(f1, encoding='latin1')
	f1.close()

	f2 = open(path + 'class_info.pickle', 'rb')
	class_info = pickle.load(f2, encoding='latin1')
	f2.close()

	f3 = open(path + 'char-CNN-RNN-embeddings.pickle', 'rb')
	embeddings = pickle.load(f3, encoding='latin1')
	embeddings = np.array(embeddings)
	f3.close()

	return embeddings, class_info, filenames
	
	
def create_bbox_map(path):
	image_path = path + 'images.txt'
	bbox_path = path + 'bounding_boxes.txt'
	
	bbox_df = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
	image_df = pd.read_csv(image_path, delim_whitespace=True, header=None)
	
	bbox_map = dict()

	for i in range(bbox_df.shape[0]):
		temp_bbox = [int(bbox_df.iloc[i][1]), int(bbox_df.iloc[i][2]), int(bbox_df.iloc[i][3]), int(bbox_df.iloc[i][4])]
		bbox_map[str(image_df.iloc[i][1])] = temp_bbox
	
	return bbox_map
	
def get_cropped_img(image_path, bbox, size):
	pr_img = Image.open(image_path).convert('RGB')
	width, height = pr_img.size

	cx = int(bbox[0] + bbox[2]/2)
	cy = int(bbox[1] + bbox[3]/2)	
	R = int(0.75 * np.maximum(bbox[2], bbox[3]))
	
	x0 = np.maximum(0, cx - R)
	y0 = np.maximum(0, cy - R)
	x1 = np.minimum(width, cx + R)
	y1 = np.minimum(height, cy + R)
	
	pt_img = pr_img.crop([x0, y0, x1, y1])

	final_img = pt_img.resize((size,size), Image.BICUBIC)
	return final_img
	
def load_epoch_dataset(aux_info_filepath, cub_dataset_filepath, image_size):
	embeddings, class_info, filenames = extract_aux_info(aux_info_filepath)
	bbox_map = create_bbox_map(cub_dataset_filepath)
	
	print(embeddings.shape)
	
	#Images
	X = []
	
	#Class Info
	y = []
	
	#Embeddings
	emb = []
	
	#Epoch Shuffle
	#np.random.shuffle(filenames)
	
	for index,filename in enumerate(filenames):
		print("Processing image", index+1)
		img_path = cub_dataset_filepath + '/images/' + filename + '.jpg'
		bbox_val = bbox_map[filename + '.jpg']
		proc_img = get_cropped_img(img_path, bbox_val, image_size)
		X.append(np.array(proc_img))
		y.append(np.array(class_info[index]))
		emb.append(np.array(embeddings[index, :, :]))
	
	return np.array(X), np.array(y), np.array(emb)

a,b,c = extract_aux_info('/home/aabhaas/Desktop/NNFL_project/birds/train/')
bbox_map = create_bbox_map('/home/aabhaas/Desktop/NNFL_project/CUB_200_2011/')
str1 = '/home/aabhaas/Desktop/NNFL_project/CUB_200_2011/images/200.Common_Yellowthroat/Common_Yellowthroat_0009_190578.jpg'

alpha = get_cropped_img(str1, bbox_map['200.Common_Yellowthroat/Common_Yellowthroat_0009_190578.jpg'], 300)
load_epoch_dataset('/home/aabhaas/Desktop/NNFL_project/birds/train/', '/home/aabhaas/Desktop/NNFL_project/CUB_200_2011/', 256)
