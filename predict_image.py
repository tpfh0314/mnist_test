import pandas as pd
from tensorflow.python.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform
import math


def predict_image(file_name):
    image = cv2.imread(file_name)

    ## 이미지 resize.
    image = imutils.resize(image,height = 500)

    ##이미지 색 변환
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    ##이미지 Blur 처리
    blurred = cv2.GaussianBlur(gray, (5,5), 0)


    ##이미지 임계처리
    ret, img_checker = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)


    ##이미지에서 숫자의 Rect 찾기

    ## 사진의 외부 범위도 객체로 탐지하기 때문에 나중에 날려버리기 위해 RETR_LIST로 설정
    contours, hierachy= cv2.findContours(img_checker.copy(),
                                     cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(each) for each in contours]

    ##숫자를 순서대로 읽기위해 Rect를 왼쪽 위부터 정렬
    rects = sorted(rects, key = lambda x : (round(x[1],-2),(round(x[0]))))

    ## 사진 외부 범위 객체로 탐지된 경우 삭제
    for rect in rects:
        if (rect[0],rect[1]) ==(0,0):
            rects.remove(rect)

    ## Rect 내부의 Rect를 제외하기 위한 List
    remove_rect_list = []

    for i, rect in enumerate(rects):
        x,y,w,h = rect

        r1_start = (x,y)
        r1_end = (x+w,y+h)

        for j, rect2 in enumerate(rects):
            if i==j:
                continue

            x, y, w, h = rect2

            r2_start = (x, y)
            r2_end = (x + w, y + h)
            ## 4, 6,8,9와 같은 숫자들은 내부에도 Rect가 생기는 문제가 발생하여 좌표 비교를 통해 내부 Rect를 삭제하는 기능입니다.
            if r1_start[0] > r2_start[0] and r1_start[1] > r2_start[1] and r1_end[0] < r2_end[0] and r1_end[1] < r2_end[1]:
                remove_rect_list.append(i)

    img_result = []
    img_for_class = image.copy()

    margin_pixel = 15


    for k,rect in enumerate(rects):
        if k in remove_rect_list: continue


        im = img_for_class[rect[1] - margin_pixel: rect[1] + rect[3] + margin_pixel,
            rect[0] - margin_pixel: rect[0] + rect[2] + margin_pixel]
        row, col = im.shape[:2]

        bord_size = max(row,col)
        diff_size = min(row,col)

        bottom = im[row-2:row,0:col]
        mean = cv2.mean(bottom)[0]

        ##이미지들의 사이즈가 전부 다르게 나오는 것을 방지하기 위해 모두 같은 사이즈로 숫자를 나눠줍니다.
        bord = cv2.copyMakeBorder(im,top = 0,bottom = 0,left = int((bord_size-diff_size)/2),
                                  right = int((bord_size - diff_size)/2),
                                  borderType = cv2.BORDER_CONSTANT,
                                  value = [mean,mean,mean])
        square = cv2.resize(bord,(28,28),interpolation = cv2.INTER_AREA)
        ##전처리 된 이미지가 저장되는 리스트.
        img_result.append(square)

    ## MNIST 손글씨 학습 모델 Load
    model = load_model('cnn_model_weight.h5')

    question_cnt = 1
    answer = pd.DataFrame(columns = ['QUESTION','ANSWER'])
    for img in img_result:

        img = img[:,:,1]
        img = img.reshape(-1,28,28,1)
        input_data = ((np.array(img)/255)-1)*-1
        res = np.argmax(model.predict(input_data),axis=-1)
        # print(res)
        answer = answer.append({'QUESTION':question_cnt,'ANSWER': res},ignore_index=True)
        question_cnt+=1

    return img_result,answer