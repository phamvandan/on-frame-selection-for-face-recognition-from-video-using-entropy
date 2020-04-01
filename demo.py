import cv2
import numpy as np
import math
import glob

def print_image(img):
    cv2.imshow("ok",img)
    cv2.waitKey(0)

def calculateEntropy(img, debug=False):
    img = cv2.resize(img, (100, 100))
    # cv2.imwrite("test1.jpg",img)
    if debug:
        print_image(img)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = hsv_img[:, :, 0]
    # if debug:
    #     print(img)
    if debug:
        cv2.imshow("ok", img)
        # cv2.imwrite("test.jpg",img)
        cv2.waitKey(0)
    # img = np.random.randint(5,size=(6,6))
    # print(img)
    (h, w) = img.shape[:2]
    ## numpy array = (row,col)

    window_size = (10, 10)

    (row, col) = window_size
    mul = row * col
    stride = 1
    entire_entropy = 0
    for i in range(0, h - row + 1):
        for j in range(0, w - col + 1):
            # print("img[",i,j,"]=",img[i,j])
            lst = [0] * 255
            entropy = 0
            for p in range(i, i + row):
                for q in range(j, j + col):
                    # print("img[",p,q,"]=",img[p,q])
                    lst[img[p, q]] = lst[img[p, q]] + 1
            for value in lst:
                if value != 0:
                    entropy = entropy + value / mul * math.log2(value / mul)
            entropy = -entropy
            # print(entropy)
            entire_entropy = entire_entropy + abs(entropy)
    if debug:
        print(entire_entropy)
    return entire_entropy


if __name__ == "__main__":
    files = glob.glob("KHANH/*.jpg")
    image_entropy = []
    for image_path in files:
        img = cv2.imread(image_path)
        entropy = calculateEntropy(img,False)
        image_entropy.append((image_path, entropy))
    image_entropy = sorted(image_entropy, key=lambda x: x[1], reverse=True)
    for t in image_entropy:
        print(t)
