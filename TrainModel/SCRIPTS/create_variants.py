#!/usr/bin/python

import sys
import cv2
import numpy as np

# See
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html


def read_image(image_location):
    gray_img = cv2.cvtColor(cv2.imread(image_location),
                            cv2.COLOR_BGR2GRAY)
    gray_img = np.array(gray_img)
    return gray_img


def left_shift_by_n(img, n=1):
    if n == 0:
        return img
    res = np.zeros(img.shape)
    res[:, -n:] = img[:, :n]
    res[:, :-n] = img[:, n:]
    return res


def right_shift_by_n(img, n=1):
    return left_shift_by_n(img, -n)


def up_shift_by_n(img, n=1):
    if n == 0:
        return img
    res = np.zeros(img.shape)
    res[-n:, :] = img[:n, :]
    res[:-n, :] = img[n:, :]
    return res


def down_shift_by_n(img, n=1):
    return up_shift_by_n(img, -n)


def blur_image(img, n=2):
    return cv2.blur(img, (n, n))


def bilateral(img):
    return cv2.bilateralFilter(img, 9, 75, 75)


def write_variants(image_location, n):
    img = read_image(image_location)
    head = image_location.split('.')[:-1]
    head = ''.join(head)
    tail = image_location.split('.')[-1]

    for k in range(1, n + 1):
        cv2.imwrite('%s_l_%d.%s' % (head, k, tail), left_shift_by_n(img, k))
        cv2.imwrite('%s_r_%d.%s' % (head, k, tail), right_shift_by_n(img, k))
        cv2.imwrite('%s_u_%d.%s' % (head, k, tail), up_shift_by_n(img, k))
        cv2.imwrite('%s_d_%d.%s' % (head, k, tail), down_shift_by_n(img, k))
    for k in range(2, 4):
        cv2.imwrite('%s_b_%d.%s' % (head, k, tail), blur_image(img, k))
    cv2.imwrite('%s_i.%s' % (head, tail), bilateral(img))


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        write_variants(filename, 3)
