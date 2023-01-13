from A1 import SVM1
from A2 import SVM2
from B1 import RF1
from B2 import RF2

import os
import dlib
global basedir, image_paths, target_size
basedir = ('.\\Datasets')
celeba_dir = os.path.join(basedir,'celeba')
images_dir = os.path.join(celeba_dir,'img')
labels_filename = 'labels.csv'

cartoon_dir = os.path.join(basedir,'cartoon_set')
cartoon_images_dir = os.path.join(cartoon_dir,'img')


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


model_A1 = SVM1()     
train_image, test_image, train_label, test_label = model_A1.data_preprocessing_A1(images_dir, celeba_dir, labels_filename)
acc_A1 = model_A1.train(train_image, test_image, train_label, test_label)


model_A2 = SVM2()     
train_image2, test_image2, train_label2, test_label2 = model_A2.data_preprocessing_A2(images_dir, celeba_dir, labels_filename)
acc_A2 = model_A2.train(train_image2, test_image2, train_label2, test_label2)

model_B1 = RF1()     
train_image3, test_image3, train_label3, test_label3 = model_B1.data_preprocessing_B1(cartoon_images_dir, cartoon_dir, labels_filename)
acc_B1 = model_B1.train(train_image3, test_image3, train_label3, test_label3)

model_B2 = RF2()     
train_image4, test_image4, train_label4, test_label4 = model_B2.data_preprocessing_B2(cartoon_images_dir, cartoon_dir, labels_filename)
acc_B2 = model_B2.train(train_image4, test_image4, train_label4, test_label4)




def print_train_test_acc(task, dct1):
	print(task + 'train accuracy: ')
	for item, value in dct1.items():
		print('{}: ({})'.format(item, value))

print_train_test_acc('Task A1', acc_A1)
print_train_test_acc('Task A2', acc_A2)      
print_train_test_acc('Task B1', acc_B1)     
print_train_test_acc('Task B2', acc_B2)        
        