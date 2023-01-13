import os
import numpy as np
from keras.preprocessing import image
import cv2
from sklearn import svm
from sklearn.svm import SVC
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

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels(images_dir, celeba_dir, labels_filename):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(celeba_dir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        wrong_img = []
        for img_path in image_paths:
            file_name= img_path.split('.')[1].split('\\') [-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(gender_labels[file_name])
            
            else:
                wrong_img.append(file_name) 

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, gender_labels, wrong_img



def get_data(X, Y):
    
    
    Y = np.array([Y, -(Y - 1)]).T
    X = X.reshape(X.shape[0], 68*2)
   
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    #x_train = x_train.reshape(x_train.shape[0], 68*2)
    #x_test = x_test.reshape(x_test.shape[0], 68*2)
    
    y_train = list(zip(*y_train))[0]
    y_test = list(zip(*y_test))[0]
    
    return x_train, x_test, y_train, y_test


def data_preprocessing_A1(images_dir, celeba_dir, labels_filename):
	# Extract feature data
	feature, label, _ = extract_features_labels(images_dir, celeba_dir, labels_filename)
	x_train, x_test, y_train, y_test = get_data(feature, label)

	return x_train, x_test, y_train, y_test



def train(X, y, test_X, test_Y):
		# Obtaining optimum hyperparameters and classifier for different kernel
	
    clf_rbf = SVC(C=10000, gamma = 0.01, kernel='rbf')
    clf_linear = SVC(C=100, kernel = 'linear', max_iter=1000000)
    clf_poly = SVC(C=1 ,degree=3, kernel = 'poly')


    clf_linear.fit(X, y)
    pred1 = clf_linear.predict(test_X)
    score1 = accuracy_score(test_Y,pred1)
    
    clf_poly.fit(X, y)
    pred2 = clf_poly.predict(test_X)
    score2 = accuracy_score(test_Y,pred2)
    
    clf_rbf.fit(X, y)
    pred3 = clf_rbf.predict(test_X)
    score3 = accuracy_score(test_Y,pred3)
    
    

    acc = {'Linear SVM': score1, 'Polynomial SVM': score2, 'RBF SVM': score3}
    

    return acc

