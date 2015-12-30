import numpy as np
import scipy as sp
import os
import sys
import glob
import shutil
import csv
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


def remove_noice(songM):
    breakpoint = 200
    songM[songM > breakpoint] = 255
    songM[songM <= breakpoint] = 0
    return songM


def find_staves(songM, OUT_FILE = 'PREPROCESSED/LINES/'):
    def project_to_left(matrix):
        vector = np.sum(matrix,axis=1)
        return vector
    
    def norm_vec(vec):
        if len(vec)==0:
            return vec
        vec = vec-min(vec)
        if max(vec) == 0:
            return vec
        return vec/float(max(vec))
        
    def smooth(vec,zahl):
        if zahl == 0:
            return vec
        vec = np.append([(2*vec[0] + vec[1])/3], [(vec[:-2] + 2*vec[1:-1] + vec[2:])/4])
        vec = np.append(vec,[(2*vec[-1] + vec[-2])/3])
        vec_smooth = smooth(vec,zahl-1)
        return vec_smooth
        
    def find_minima(vec):
        indices_minima = argrelextrema(vec, np.greater)
        return indices_minima
    
    def get_corepart(vec, index_left, index_right):
        corepart_number = 0.99
        part_vec = vec[index_left:index_right]
        if len(np.where(part_vec < corepart_number*part_vec[0])[0]):
            index_left = np.where(part_vec < corepart_number*part_vec[0])[0][0] + index_left
        if len(np.where(part_vec < corepart_number*part_vec[-1])[0]):
            index_right = index_right - len(part_vec) + np.where(part_vec < corepart_number*part_vec[-1])[0][-1]
        return index_left, index_right     
        
    def check_if_stave(part):
        translate = 0.4
        vec = project_to_left(part)
        vec = norm_vec(vec) - translate
        #plt.plot(vec)
        #plt.show()
        num = 0
        for zahl in range(len(vec)-1):
            if vec[zahl] >= 0 and vec[zahl+1] < 0:
                num = num + 1
            if vec[zahl] <= 0 and vec[zahl+1] > 0:
                num = num + 1
        return num >= 8
        
    def separate_to_staves(matrix, vec, indices_minima):
        index_left = indices_minima[0]
        for zahl,index_right in enumerate(indices_minima[1:]):
            index_left, index_right = get_corepart(vec, index_left, index_right)
            part = matrix[index_left:index_right,:]
            if check_if_stave(part):
                num_files = len(glob.glob(OUT_FILE + '*.jpg'))
                sp.misc.imsave(OUT_FILE + 'line%03i.jpg' % num_files, part)
            index_left = index_right
    
    tryer = project_to_left(songM)
    tryer = norm_vec(tryer)
    #plt.plot(tryer)
    
    how_oft_smooth = 200
    tryer = smooth(tryer,how_oft_smooth)
    tryer = norm_vec(tryer)
    #plt.plot(tryer)
    #plt.show()
    
    index_minima = np.append(0,np.asarray(find_minima(tryer))[0])
    separate_to_staves(songM, tryer, index_minima)   


if __name__ == '__main__':
    list_songs = sys.argv[2:]
    OUT_FILE = sys.argv[1]
    for song_file in sorted(list_songs):
        song = Image.open(song_file).convert('L')
        songM = np.array(song)
        #songM = remove_noice(songM)
        find_staves(songM, OUT_FILE = OUT_FILE)

