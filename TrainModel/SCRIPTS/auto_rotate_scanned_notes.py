#!/usr/bin/python

import sys
import cv2
import numpy as np
import scipy.optimize


def auto_rotate_scanned_notes(image_location, OUT_FILE='out.jpg', DEBUG=False):
    gray_img = cv2.cvtColor(cv2.imread(image_location),
                            cv2.COLOR_BGR2GRAY)
    gray_img = np.array(gray_img)

    rows, cols = gray_img.shape[:2]

    def rotate_by_angle(img, angle):
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    LEFT_CUTOFF = int(rows * 0.1)
    RIGHT_CUTOFF = int(rows * 0.9)

    def find_best_angle(img):
        '''
        Simply run through all reasonable angles
        Could be implemented as a typical find_min
        '''
        running_min_row = float('Inf')
        running_best_angle = 0.0
        for angle in np.linspace(-2, 2, 500):
            rotated = rotate_by_angle(img, angle)[LEFT_CUTOFF:RIGHT_CUTOFF]
            min_row = min(np.sum(rotated, axis=1))
            if min_row < running_min_row:
                running_min_row = min_row
                running_best_angle = angle
        return running_best_angle

    def find_best_angle_2(img):
        '''
        Uses scipy.optimize
        '''
        def fun_to_min(angle):
            #            if DEBUG:
            #                print(angle)
            rotated = rotate_by_angle(img, angle)[LEFT_CUTOFF:RIGHT_CUTOFF]
            return min(np.sum(rotated, axis=1))

        return scipy.optimize.minimize_scalar(fun_to_min).x

    rotation_angle = find_best_angle(gray_img)
    res = rotate_by_angle(gray_img, rotation_angle)
    if DEBUG:
        print(rotation_angle)
    cv2.imwrite(OUT_FILE, res)

if __name__ == '__main__':
    n_args = len(sys.argv) - 1
    out_str = ["%s/noten_gerade_%d.jpg" %
               (sys.argv[1], n + 1) for n in range(n_args)]
    for IN, OUT in zip(sys.argv[2:], out_str):
        auto_rotate_scanned_notes(IN, OUT_FILE=OUT, DEBUG=True)
