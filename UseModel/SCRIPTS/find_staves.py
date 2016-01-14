import numpy as np
import scipy as sp
import scipy.misc
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


def find_staves(songM, OUT_FILE = 'PREPROCESSED/STAVES/'):
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
        vec_smooth = np.append([(2*vec[0] + vec[1])/3.0], [(vec[:-2] + 2*vec[1:-1] + vec[2:])/4.0])
        vec_smooth = np.append(vec_smooth,[(2*vec[-1] + vec[-2])/3.0])
        vec_smooth = smooth(vec_smooth,zahl-1)
        return vec_smooth
        
    def find_minima(vec):
        indices_minima = argrelextrema(vec, np.greater)
        return indices_minima
    
    def get_corepart_horizontal(vec, index_left, index_right):
        corepart_number = 0.99
        part_vec = vec[index_left:index_right]
        if len(np.where(part_vec < corepart_number*part_vec[0])[0]):
            index_left = np.where(part_vec < corepart_number*part_vec[0])[0][0] + index_left
        if len(np.where(part_vec < corepart_number*part_vec[-1])[0]):
            index_right = index_right - len(part_vec) + np.where(part_vec < corepart_number*part_vec[-1])[0][-1]
        return index_left, index_right     
        
    def get_corepart_vertical(part):
        translate = 0.7
        add_number = 3
        smooth_number = 3
        i = 0
        continu = True
        while continu:
            vec = np.sum(part[:,i:i+add_number],axis=1)/add_number
            vec = smooth(vec,smooth_number)/255.0 - translate
            num = 0
            for zahl in range(len(vec)-1):
                if vec[zahl] >= 0 and vec[zahl+1] < 0:
                    num = num + 1
                if vec[zahl] <= 0 and vec[zahl+1] > 0:
                    num = num + 1
            continu = num < 6  and i < 300      
            i=i+add_number
        indexleft = i-add_number*2

        i = part.shape[1]
        continu = True
        while continu:
            vec = np.sum(part[:,i-add_number:i],axis=1)/add_number 
            vec = smooth(vec,smooth_number)/255.0 - translate
            #plt.plot(vec)
            num = 0
            for zahl in range(len(vec)-1):
                if vec[zahl] >= 0 and vec[zahl+1] < 0:
                    num = num + 1
                if vec[zahl] <= 0 and vec[zahl+1] > 0:
                    num = num + 1
            #print num
            #plt.show()
            continu = num < 6 and i > 700
            i=i-add_number
        indexright = i+add_number*2
        return part[:,indexleft:indexright]

    def check_if_stave(part):
        translate = 0.45
        vec = project_to_left(part)
        vec = norm_vec(vec) - translate
        #plt.plot(vec)
        num = 0
        for zahl in range(len(vec)-1):
            if vec[zahl] >= 0 and vec[zahl+1] < 0:
                num = num + 1
            if vec[zahl] <= 0 and vec[zahl+1] > 0:
                num = num + 1
        #print num
        #plt.show()
        return num >= 8
        
    def separate_to_staves(matrix, vec, indices_minima):
        index_left = indices_minima[0]
        for zahl,index_right in enumerate(indices_minima[1:]):
            index_left, index_right = get_corepart_horizontal(vec, index_left, index_right)
            part = matrix[index_left:index_right,:]
            if check_if_stave(part):
                part = get_corepart_vertical(part)
                if check_if_stave(part):
                    num_files = len(glob.glob(OUT_FILE + '*.jpg'))
                    scipy.misc.imsave(OUT_FILE + 'line%03i.jpg' % num_files, part)
            index_left = index_right
    
    tryer = project_to_left(songM)
    tryer = norm_vec(tryer)
    #plt.plot(tryer, range(len(tryer)))
    
    how_oft_smooth = 200
    tryer = smooth(tryer,how_oft_smooth)
    tryer = norm_vec(tryer)
    #plt.plot(tryer, range(len(tryer)))
    #plt.savefig('im4.jpg', facecolor='w', edgecolor='w')
    #plt.show()
    
    
    index_minima = np.append(0,np.asarray(find_minima(tryer))[0])
    index_minima = np.append(index_minima,len(tryer)-1)
    separate_to_staves(songM, tryer, index_minima)   


if __name__ == '__main__':
    list_songs = sys.argv[2:]
    OUT_FILE = sys.argv[1]
    for song_file in sorted(list_songs):
        song = Image.open(song_file).convert('L')
        songM = np.array(song)
        #songM = remove_noice(songM)
        find_staves(songM, OUT_FILE = OUT_FILE)

