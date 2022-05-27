import os
import random
import math
import shutil

def split_data(src_directory, dest_directory, train_percentage, classes_file):
	for img_class in os.listdir(src_directory):
		class_directory_first_level = os.path.normpath(os.path.join(src_directory, img_class, '0'))
		cases_in_class = os.listdir(class_directory_first_level)
		random.shuffle(cases_in_class)
		split_point = int(math.ceil(len(cases_in_class) * (1 - train_percentage)))
		train_cases = cases_in_class[0:split_point]
		test_cases = cases_in_class[split_point + 1:]
		copy_images_to_dest(img_class, os.path.normpath(os.path.join(src_directory, img_class)), train_cases, os.path.normpath(os.path.join(dest_directory, "train")))
		copy_images_to_dest(img_class, os.path.normpath(os.path.join(src_directory, img_class)), test_cases, os.path.normpath(os.path.join(dest_directory, "test")))

def copy_images_to_dest(img_class, class_directory, cases, dest_directory):
	for case in cases:
		for i in range(6):
			source_directory = os.path.normpath(os.path.join(class_directory, str(i), case))
			idx = 0
			if (os.path.exists(source_directory)):
				for image in os.listdir(source_directory):
					image_path = os.path.normpath(os.path.join(source_directory, image))
					image_dest = os.path.normpath(os.path.join(dest_directory, img_class, case + "_" + str(i) + "_" + str(idx) + ".jpg"))

					if not os.path.exists(os.path.normpath(os.path.join(dest_directory, img_class))):
						os.makedirs(os.path.normpath(os.path.join(dest_directory, img_class)))

					shutil.copyfile(image_path, image_dest)
					idx = idx + 1
				
split_data('E:/dataset', 'E:/split_dataset', 0.2, 'E:/split_dataset/classes.txt')

