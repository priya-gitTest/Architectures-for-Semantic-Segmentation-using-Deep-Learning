from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2

from PIL import Image


class dataAugmentation(object):
	
	"""
	A class used to augment image
	Firstly, read train image and label seperately, and then merge them together for the next process
	Secondly, use keras preprocessing to augmentate image
	Finally, seperate augmentated image apart into train image and label
	"""

	def __init__(self, train_path="./data/train/image", label_path="./data/train/label",
				 merge_path="./data/train/merge", aug_merge_path="./data/train/aug_merge", 
				 aug_train_path="./data/train/aug_images", 
				 aug_label_path="./data/train/aug_masks", img_type="tif"):
		
		"""
		Using glob to get all .img_type form path

		"""

		self.train_imgs = glob.glob(train_path+"/*."+img_type)
		self.label_imgs = glob.glob(label_path+"/*."+img_type)
		self.train_path = train_path
		self.label_path = label_path
		self.merge_path = merge_path
		self.img_type = img_type
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path

		if not os.path.exists(merge_path):
			os.mkdir(merge_path)
			os.mkdir(aug_merge_path)
			os.mkdir(aug_train_path)
			os.mkdir(aug_label_path)

		self.slices = len(self.train_imgs)
		self.datagen = ImageDataGenerator(
									preprocessing_function=self.preprocess,
									rotation_range=0.2,
									width_shift_range=0.1,
									height_shift_range=0.1,
									shear_range=0.05,
									zoom_range=0.05,
									horizontal_flip=True,
									fill_mode='nearest')

	def quad_as_rect(self, quad):
	    if quad[0] != quad[2]: return False
	    if quad[1] != quad[7]: return False
	    if quad[4] != quad[6]: return False
	    if quad[3] != quad[5]: return False
	    return True

	def quad_to_rect(self, quad):
	    assert(len(quad) == 8)
	    assert(self.quad_as_rect(quad))
	    return (quad[0], quad[1], quad[4], quad[3])

	def rect_to_quad(self, rect):
	    assert(len(rect) == 4)
	    return (rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1])

	def shape_to_rect(self, shape):
	    assert(len(shape) == 2)
	    return (0, 0, shape[0], shape[1])

	def griddify(self, rect, w_div, h_div):
	    w = rect[2] - rect[0]
	    h = rect[3] - rect[1]
	    x_step = w / float(w_div)
	    y_step = h / float(h_div)
	    y = rect[1]
	    grid_vertex_matrix = []
	    for _ in range(h_div + 1):
	        grid_vertex_matrix.append([])
	        x = rect[0]
	        for _ in range(w_div + 1):
	            grid_vertex_matrix[-1].append([int(x), int(y)])
	            x += x_step
	        y += y_step
	    grid = np.array(grid_vertex_matrix)
	    return grid

	def distort_grid(self, org_grid, max_shift):
	    new_grid = np.copy(org_grid)
	    x_min = np.min(new_grid[:, :, 0])
	    y_min = np.min(new_grid[:, :, 1])
	    x_max = np.max(new_grid[:, :, 0])
	    y_max = np.max(new_grid[:, :, 1])
	    new_grid += np.random.randint(- max_shift, max_shift + 1, new_grid.shape)
	    new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
	    new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
	    new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
	    new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
	    return new_grid

	def grid_to_mesh(self, src_grid, dst_grid):
	    assert(src_grid.shape == dst_grid.shape)
	    mesh = []
	    for i in range(src_grid.shape[0] - 1):
	        for j in range(src_grid.shape[1] - 1):
	            src_quad = [src_grid[i    , j    , 0], src_grid[i    , j    , 1],
	                        src_grid[i + 1, j    , 0], src_grid[i + 1, j    , 1],
	                        src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
	                        src_grid[i    , j + 1, 0], src_grid[i    , j + 1, 1]]
	            dst_quad = [dst_grid[i    , j    , 0], dst_grid[i    , j    , 1],
	                        dst_grid[i + 1, j    , 0], dst_grid[i + 1, j    , 1],
	                        dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
	                        dst_grid[i    , j + 1, 0], dst_grid[i    , j + 1, 1]]
	            dst_rect = self.quad_to_rect(dst_quad)
	            mesh.append([dst_rect, src_quad])
	    return mesh

	def preprocess(self, im):
		im = array_to_img(im)
		dst_grid = self.griddify(self.shape_to_rect(im.size), 3, 3)
		src_grid = self.distort_grid(dst_grid, 10)
		mesh = self.grid_to_mesh(src_grid, dst_grid)
		im = im.transform(im.size, Image.MESH, mesh)
		return img_to_array(im)

	def Augmentation(self):

		"""
		Start augmentation.....
		"""
		trains = self.train_imgs
		labels = self.label_imgs
		path_train = self.train_path
		path_label = self.label_path
		path_merge = self.merge_path
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path

		if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
			print("trains can't match labels")
			return 0

		for i in range(len(trains)):
			img_t = load_img(path_train+"/"+str(i)+"."+imgtype)
			img_l = load_img(path_label+"/"+str(i)+"."+imgtype)
			x_t = img_to_array(img_t)
			x_l = img_to_array(img_l)
			x_t[:,:,2] = x_l[:,:,0]
			img_tmp = array_to_img(x_t)
			img_tmp.save(path_merge+"/"+str(i)+"."+imgtype)
			img = x_t
			img = img.reshape((1,) + img.shape)
			savedir = path_aug_merge + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			self.augment(img, savedir, str(i))


	def augment(self, img, save_to_dir, save_prefix, batch_size=1, save_format='png', imgnum=50):
		"""
		augmentate one image

		"""

		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
						  batch_size=batch_size,
						  save_to_dir=save_to_dir,
						  save_prefix=save_prefix,
						  save_format=save_format):
			i += 1
			if i > imgnum:
				break

	def splitMerge(self):

		"""
		split merged image apart
		"""
		path_merge = self.aug_merge_path
		path_train = self.aug_train_path
		path_label = self.aug_label_path
		for i in range(self.slices):
			path = path_merge + "/" + str(i)
			# print(path)
			train_imgs = glob.glob(path+"/*."+self.img_type)
			# print(len(train_imgs))
			# break
			for imgname in train_imgs:
				midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
				img = cv2.imread(imgname)
				img_train = img[:,:,2]#cv2 read image rgb->bgr
				img_label = img[:,:,0]
				cv2.imwrite(path_train+"/"+midname+"."+self.img_type,img_train)
				cv2.imwrite(path_label+"/"+midname+"."+self.img_type,img_label)


class dataProcess(object):

	def __init__(self, out_rows, out_cols, data_path = "../data/120x160/u-net/train/aug_images", 
				 label_path = "../data/120x160/u-net/train/aug_masks", 
				 test_path = "../data/120x160/u-net/test/images", 
				 test_label_path = "../data/120x160/u-net/test/masks",
				 npy_path = "../data/120x160/u-net/npydata", img_type = "png"):

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.test_label_path = test_label_path
		self.npy_path = npy_path

		if not os.path.exists(npy_path):
			os.mkdir(npy_path)


	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.data_path + "/" + midname,grayscale = True)
			label = load_img(self.label_path + "/" + midname,grayscale = True)
			img = img_to_array(img)
			label = img_to_array(label)
			
			imgdatas[i] = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')


	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.test_path + "/" + midname,grayscale = True)
			img = img_to_array(img)
			
			imgdatas[i] = img
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')


	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255

		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255

		return imgs_test

if __name__ == "__main__":

	aug = dataAugmentation()
	aug.Augmentation()
	aug.splitMerge()
	mydata = dataProcess(512, 512)
	mydata.create_train_data()
	mydata.create_test_data()
