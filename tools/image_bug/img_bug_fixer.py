import glob

import cv2
import numpy as np


def find_shift(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    shift = cv2.phaseCorrelate(np.float32(g1), np.float32(g2))
    return shift


sample = 123
frame = 117
img_dir = f'./dataset_pending/{sample}/**/'
img_good_name = f'{sample}_0_{frame - 1}.png'
img_bad_name = f'{sample}_0_{frame}.png'
img_good_path = glob.glob(img_dir + img_good_name)[0]
img_bad_path = glob.glob(img_dir + img_bad_name)[0]

img_good = cv2.imread(img_good_path)
img_bad = cv2.imread(img_bad_path)

(dx, dy), _ = find_shift(img_good, img_bad)
print(f"dx={dx}, dy={dy}")

fixed_img = np.roll(img_bad, int(-dx), axis=1)
cv2.imshow('Image', img_good)
cv2.waitKey(1000)
cv2.imshow('Image', img_bad)
cv2.waitKey(1000)
cv2.imshow('Image', fixed_img)
cv2.waitKey(100000)

cv2.destroyAllWindows()
