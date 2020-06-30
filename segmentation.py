import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_indicator(
    img: '(np.array ==> gray scale image) white is for B-lite. black is for others',
    kernel_size: '(int or tuple[int, int]) size of the blur filter (in this case mean filter) Ex. 3 means using a 3x3 mean filter'=41,
    ratio: '(float) betweeen 0. to 1. ==> the threshold used after mean filtering'=0.25,
    area_thres: '(tuple[int, int]) ignore all areas which is not inside this range'=(200, 3000)
):
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    blur = cv2.blur(img, kernel_size)
    # remove padding
    blur = np.copy(blur[kernel_size[0]//2 + 1: -kernel_size[0]//2 + 1,kernel_size[1]//2 + 1: -(kernel_size[1]//2) + 1])
    thresh = int(ratio*255)
    _, out = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(np.uint8(out), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    tmp_area = {}

    try:
        if hierarchy == None:
            return 0, 0, 0
    except:
        for ind, tmp_hi in enumerate(hierarchy[0]):
            if tmp_hi[3] == -1:
                tmp_area[ind] = cv2.contourArea(contours[ind])
        for ind, tmp_hi in enumerate(hierarchy[0]):
            if tmp_hi[3] != -1:
                tmp_area[tmp_hi[3]] -= cv2.contourArea(contours[ind])
        tmp_area = [tmp_area[area_id] for area_id in tmp_area if area_thres[1] > tmp_area[area_id] > area_thres[0]]
        plt.hist(tmp_area, 50)
        plt.show()
        N = len(tmp_area)

        percentage = np.sum(out/255)/(out.shape[0]*out.shape[1])
        print(f'Number of contours: {N}')
        print(f'Percentage( % ): {100*percentage}')
        print(f'Index: {percentage*N}')
        return N, 100*percentage, percentage*N