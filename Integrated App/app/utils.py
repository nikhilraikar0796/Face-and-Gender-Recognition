import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
import cv2

## Loading the Models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
mean = pickle.load(open('./model/mean_preprocess.pickle' , 'rb'))
model_svm = pickle.load(open('./model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('./model/pca_50.pickle', 'rb'))

##Setting
gender_pre = ['Male' , 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX


def pipeline_model(path , filename , color="bgr"):
    ## Step 1 : Read the image
    img = cv2.imread(path)
    ## Step 2 : Convert into gray scale
    if color == 'bgr':
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)

    ## Step 3 : Crop the face
    faces = haar.detectMultiScale(gray , 1.5 , 5)
    for x,y,w,h in faces:
        cv2.rectangle(img , (x,y) , (x+w , y+h) , (255,255,0) , 2)
        roi = gray[y:y+h , x:x+w]
        ## Step 4 : Normalization
        roi = roi / 255.0
        ## Step 5 : Resize images
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi , (100,100) , cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi , (100,100) , cv2.INTER_CUBIC)

        ### Step 6 : Flattening
        roi_reshape = roi_resize.reshape(1,10000) ## 1 x 10000

        ## Step 7 : Subtract from mean
        roi_mean = roi_reshape - mean

        ## Step 8 : Get Eigen Image
        eigen_image = model_pca.transform(roi_mean)

        ## Step 9 : Pass to ML model
        results = model_svm.predict_proba(eigen_image)[0]

        ## Step 10 : 
        predict = results.argmax()
        score = results[predict]

        ## Step 11 :
        text = "%s : %0.2f "%(gender_pre[predict] , score)
        cv2.putText(img , text , (x,y) , font , 1 , (255,255,0), 2)
    
    cv2.imwrite('./static/predict/{}'.format(filename) , img)