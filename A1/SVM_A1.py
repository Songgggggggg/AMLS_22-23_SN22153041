import os
import numpy as np
import cv2
import dlib
import pandas as pd
from sklearn.svm import SVC

from keras.preprocessing import image
#from keras.utils import image_utils as image

global basedir, image_paths, target_size
basedir = ('.\\Datasets')
celeba_dir = os.path.join(basedir,'celeba')
images_dir = os.path.join(celeba_dir,'img')
labels_filename = 'labels.csv'

#predictor_path = "C:/Users/Guosheng Song/Desktop/AML/NN/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

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

def extract_features_labels():
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

feature, label, wrong_img = extract_features_labels()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#training_images, training_labels, test_images, test_labels = get_data()

def get_data_val(X, Y):
    
    
    Y = np.array([Y, -(Y - 1)]).T
    #X, Y = shuffle(X, Y)
    X = X.reshape(X.shape[0], 68*2)
    scaler.fit(X)
    X = scaler.transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25, random_state = 42)
    #x_train = x_train.reshape(x_train.shape[0], 68*2)
    #x_test = x_test.reshape(x_test.shape[0], 68*2)
    #x_val = x_val.reshape(x_val.shape[0], 68*2)
    
    y_train = list(zip(*y_train))[0]
    y_test = list(zip(*y_test))[0]
    y_val = list(zip(*y_val))[0]
    return x_train, x_test, x_val, y_train, y_test, y_val

def get_data(X, Y):
    
    
    Y = np.array([Y, -(Y - 1)]).T
    X = X.reshape(X.shape[0], 68*2)
    scaler.fit(X)
    X = scaler.transform(X)
    #X, Y = shuffle(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    #x_train = x_train.reshape(x_train.shape[0], 68*2)
    #x_test = x_test.reshape(x_test.shape[0], 68*2)
    
    y_train = list(zip(*y_train))[0]
    y_test = list(zip(*y_test))[0]
    
    return x_train, x_test, y_train, y_test


train_image, test_image, val_image, train_label, test_label, val_label = get_data_val(feature, label)
#train_image, test_image, train_label, test_label = get_data(feature, label)


"""

from sklearn.model_selection import RandomizedSearchCV
svc = SVC()

parameters = {'kernel': ['rbf','sigmoid','poly', 'linear'],
               'C': [0.1, 1.0, 10.0, 100.0, 1000.0,],
               'gamma': [ 0.001, 0.01, 0.1, 1.0],
             }

clf = RandomizedSearchCV(estimator = svc, param_distributions = parameters, cv = 5)
clf.fit(train_image, train_label)

print("Best: %f using %s" % (clf.best_score_, clf.best_params_))

"""





from sklearn.model_selection import GridSearchCV
import time


parameters = [
    {
        'kernel': ['rbf'],
        'C': [1 * 10**i for i in range(-3, 10)],
        'gamma': [1 * 10**i for i in range(-3, 6)],
        'class_weight': ['balanced'] 
    }
]

clf = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=5, n_jobs=5, scoring='f1_macro')
start = time.time()
clf.fit(train_image, train_label)
elapsed = time.time() - start
print("Fitting finished in %d min %d s" % (elapsed / 60, elapsed % 60))
print("Best set score:{:.2f}".format(clf.best_score_))
print("Best parameters:{}".format(clf.best_params_))
print("validation set score:{:.2f}".format(clf.score(val_image, val_label)))


parameters = [
    {
        'kernel': ['linear'],   # 线性核函数
        'C': [1 * 10**i for i in range(-3,11)],
        'class_weight': ['balanced'], # 样本均衡度
        'max_iter': [1000000]   # 限制最多迭代1000000次
    },
]

clf = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=5, n_jobs=5, scoring='f1_macro')
start = time.time()
clf.fit(train_image, train_label)
elapsed = time.time() - start
print("Fitting finished in %d min %d s" % (elapsed / 60, elapsed % 60))
print("Best set score:{:.2f}".format(clf.best_score_))
print("Best parameters:{}".format(clf.best_params_))
print("Test set score:{:.2f}".format(clf.score(val_image, val_label)))


parameters = [
    {
        'kernel': ['poly'],		# 多项式核函数
        'C': [1 * 10**i for i in range(-3, 11)],
        'degree': range(2, 10),
        'class_weight': ['balanced'],  #样本均衡度
        #'max_iter': [1000000]	# 限制最多迭代1000000次
    }
]

clf = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=5, n_jobs=5, scoring='f1_macro')
start = time.time()
clf.fit(train_image, train_label)
elapsed = time.time() - start
print("Fitting finished in %d min %d s" % (elapsed / 60, elapsed % 60))
print("Best set score:{:.2f}".format(clf.best_score_))
print("Best parameters:{}".format(clf.best_params_))
print("Test set score:{:.2f}".format(clf.score(val_image, val_label)))




import matplotlib.pyplot as plt
from sklearn.model_selection  import learning_curve
from sklearn.model_selection  import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    #if axes is None:
        #_, axes = plt.subplots(1, 3, figsize=(20, 5))

    #axes[0].set_title(title)
    plt.Figure(figsize=(10,8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y,
                                                                          cv=cv, n_jobs=n_jobs,
                                                                          train_sizes=train_sizes,
                                                                          return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    #fit_times_mean = np.mean(fit_times, axis=1)
    #fit_times_std = np.std(fit_times, axis=1)
    
    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")
    """
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    """

    return plt

cv = cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)


clf_rbf = SVC(C=10000, gamma = 0.01, kernel='rbf')
clf_linear = SVC(C=100, kernel = 'linear', max_iter=1000000)
clf_poly = SVC(C=1 ,degree=3, kernel = 'poly')

title  = "Learning Curve (RBF kernel)"
plot_learning_curve(clf_rbf, title, train_image, train_label, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title  = "Learning Curve (linear kernel)"
plot_learning_curve(clf_linear, title, train_image, train_label, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title  = "Learning Curve (poly kernel)"
plot_learning_curve(clf_poly, title, train_image, train_label, ylim=(0.7, 1.01), cv=cv, n_jobs=4)



from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pandas as pd
clf_rbf.fit(train_image, train_label)
rbf_pred = clf_rbf.predict(test_image)



clf_linear.fit(train_image, train_label)
linear_pred = clf_linear.predict(test_image)


clf_poly.fit(train_image, train_label)
poly_pred = clf_poly.predict(test_image)

Performance_metrics = {'Algorithm': ['SVC RBF','SVC LINEAR','SVC POLY'],

              'Precision' : [precision_score(test_label, rbf_pred, average="micro"),
                                   precision_score(test_label, linear_pred, average="micro"),
                                   precision_score(test_label, poly_pred, average="micro")],
              'Recall' : [recall_score(test_label, rbf_pred, average="micro"),
                                recall_score(test_label, linear_pred, average="micro"),
                                recall_score(test_label, poly_pred, average="micro")],
              'F1 Score': [f1_score(test_label, rbf_pred, average="micro"),
                           f1_score(test_label, linear_pred, average="micro"),
                           f1_score(test_label, poly_pred, average="micro")]}

PM = pd.DataFrame(Performance_metrics, columns = ['Algorithm', 'Precision ', 'Recall ' , 'F1 Score'])

print(Performance_metrics)

print(classification_report(test_label,rbf_pred))
print(classification_report(test_label,linear_pred))
print(classification_report(test_label,poly_pred))


print(precision_score(test_label, rbf_pred, average="micro"))
print(precision_score(test_label, linear_pred, average="micro"))
print(precision_score(test_label, poly_pred, average="micro"))

print(recall_score(test_label, rbf_pred, average="micro"))
print(recall_score(test_label, linear_pred, average="micro"))
print(recall_score(test_label, poly_pred, average="micro"))

print(f1_score(test_label, rbf_pred, average="micro"))
print(f1_score(test_label, linear_pred, average="micro"))
print(f1_score(test_label, poly_pred, average="micro"))

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm1 = confusion_matrix(test_label,rbf_pred)

fig, ax = plt.subplots(figsize=(5,5))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm1, annot=True,fmt="d", cmap='YlGnBu', linecolor='black', linewidths=1)
ax.set_title('SVM_RBF');
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted');
ax.xaxis.set_ticklabels(['Male', 'Female'])
ax.yaxis.set_ticklabels(['Male', 'Female'])


cm2 = confusion_matrix(test_label,linear_pred)

fig, ax = plt.subplots(figsize=(5,5))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm2, annot=True,fmt="d", cmap='OrRd', linecolor='black', linewidths=1)
ax.set_title('SVM_LINEAR');
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted');
ax.xaxis.set_ticklabels(['Male', 'Female'])
ax.yaxis.set_ticklabels(['Male', 'Female'])

cm3 = confusion_matrix(test_label,poly_pred)

fig, ax = plt.subplots(figsize=(5,5))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm3, annot=True,fmt="d", cmap='Purples', linecolor='black', linewidths=1)
ax.set_title('SVM_POLY');
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted');
ax.xaxis.set_ticklabels(['Male', 'Female'])
ax.yaxis.set_ticklabels(['Male', 'Female'])
