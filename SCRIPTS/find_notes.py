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


def find_notes(lineM):
    def project_to_bottom(matrix):
        vector = np.sum(matrix,axis=0)
        return vector
    
    def norm_vec(vector):
        vector = vector-min(vector)
        if max(vector) == 0:
            return vector
        return vector/float(max(vector))
        
    def smooth(vec,zahl):
        if zahl == 0:
            return vec
        vec = np.append([(2*vec[0] + vec[1])/3], [(vec[:-2] + 2*vec[1:-1] + vec[2:])/4])
        vec = np.append(vec,[(2*vec[-1] + vec[-2])/3])
        vec_smooth = smooth(vec,zahl-1)
        return vec_smooth
        
    def find_minima(vector):
        indices_minima = argrelextrema(vector, np.greater)
        return indices_minima
    
    def separate_to_notes(matrix, indices_minima):
        index_left = indices_minima[0]
        parts = []
        for zahl,index_right in enumerate(indices_minima[1:]):
            part = matrix[:,index_left:index_right]
            index_left = index_right
            parts.append(part)
        return parts
    
    tryer = project_to_bottom(lineM)
    tryer = norm_vec(tryer)
    #plt.plot(tryer)
    
    how_oft_smooth = 20
    tryer = smooth(tryer,how_oft_smooth)
    tryer = norm_vec(tryer)
    #plt.plot(tryer)
    #plt.savefig('im8.jpg', facecolor='w', edgecolor='w')
    #plt.show()
    
    index_minima = np.append(0,np.asarray(find_minima(tryer))[0])
    parts = separate_to_notes(lineM,index_minima)
    return parts
    


def equalize(note):
    width = 32
    hight = 80
    
    def ensure_filesize(matrix):
        h_plus = 255*np.ones((hight/2,matrix.shape[1]))
        matrix = np.append(h_plus,matrix,axis=0)
        matrix = np.append(matrix,h_plus,axis=0)
        v_plus = 255*np.ones((matrix.shape[0],width/2))
        matrix = np.append(v_plus,matrix,axis=1)
        matrix = np.append(matrix,v_plus,axis=1)
        return matrix
    
    def shrink_to_size(matrix):
        index_low = matrix.shape[0]/2 - hight/2
        index_up = matrix.shape[0]/2 + hight/2
        index_left = matrix.shape[1]/2 - width/2
        index_right = matrix.shape[1]/2 + width/2        
        return matrix[index_low:index_up,index_left:index_right]
        
    note = ensure_filesize(note)
    note = shrink_to_size(note)
    return note
    

if __name__ == '__main__':
    list_lines = sys.argv[2:]
    OUT_FILE = sys.argv[1]
    for line_file in sorted(list_lines):
        line = Image.open(line_file).convert('L')
        lineM = np.array(line)
        parts = find_notes(lineM)
        for note in parts:
            note = equalize(note)
            num_files = len(glob.glob(OUT_FILE + '*.jpg'))
            sp.misc.imsave(OUT_FILE + 'note1%04i.jpg' % num_files, note)
        

