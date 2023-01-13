import os
import numpy as np
from keras.preprocessing import image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def crop(img): 
    crop = img[60:75 , 40:88]
    return crop



def extract_features_labels_pixel(cartoon_images_dir, cartoon_dir, labels_filename):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(cartoon_images_dir, l) for l in os.listdir(cartoon_images_dir)]
    target_size = (128,128)
    labels_file = open(os.path.join(cartoon_images_dir, labels_filename), 'r')
    lines = labels_file.readlines()
    eyecolor_labels = {line.split('\t')[0] : int(line.split('\t')[1]) for line in lines[1:]}
    if os.path.isdir(cartoon_images_dir):
        all_features = []
        all_labels = []

        for img_path in image_paths:
            file_name= img_path.split('.')[1].split('\\') [-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            img_crop = crop(img)
            
            all_features.append(img_crop)
            all_labels.append(eyecolor_labels[file_name])
            

    eyecolor_features = np.array(all_features)
    faceshape_labels = np.array(all_labels)
    #gender_labels = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    return  eyecolor_features, faceshape_labels



def get_data(X, Y):
    
    
    Y = np.array([Y, -(Y - 1)]).T
    X = X.reshape(X.shape[0], 16*2)
   
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    #x_train = x_train.reshape(x_train.shape[0], 68*2)
    #x_test = x_test.reshape(x_test.shape[0], 68*2)
    
    y_train = list(zip(*y_train))[0]
    y_test = list(zip(*y_test))[0]
    
    return x_train, x_test, y_train, y_test


def data_preprocessing_B2(cartoon_images_dir, cartoon_dir, labels_filename):
	# Extract feature data
	feature, label, _ = extract_features_labels_pixel(cartoon_images_dir, cartoon_dir, labels_filename)
	x_train, x_test, y_train, y_test = get_data(feature, label)

	return x_train, x_test, y_train, y_test



def train(X, y, test_X, test_Y):
		# Obtaining optimum hyperparameters and classifier for different kernel
	
    best_para_rf = RandomForestClassifier(max_features=0.2,
                                      max_depth=15,
                                      max_samples=0.75,
                                      n_estimators=110)

    best_para_rf.fit(X, y)
    rf_pred = best_para_rf.predict(test_X)
    score = accuracy_score(test_Y,rf_pred)
   

    acc = {'RF': score}
    

    return acc

