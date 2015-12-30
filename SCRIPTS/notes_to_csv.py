import numpy as np
import csv
import glob
import sys
import os
from PIL import Image


def folder_to_csv_min(folder, Xwriter, Ywriter, y, minimum):
    for note_file in sorted(glob.glob(folder + '/*.jpg'))[:minimum]:
        note = Image.open(note_file)
        note = np.asarray(note)
        to_save = np.reshape(note,(1,note.shape[0]*note.shape[1]))
        Xwriter.writerows(to_save)
        Y = np.reshape(np.array(y),(1,1))   
        Ywriter.writerows(Y)

def folder_to_csv(folder, Xwriter, Ywriter, y):
    for note_file in sorted(glob.glob(folder + '/*.jpg')):
        note = Image.open(note_file)
        note = np.asarray(note)
        to_save = np.reshape(note,(1,note.shape[0]*note.shape[1]))
        Xwriter.writerows(to_save)
        Y = np.reshape(np.array(y),(1,1))   
        Ywriter.writerows(Y)

            
if __name__ == '__main__':
    OUT_FILE_X = sys.argv[1]
    OUT_FILE_Y = sys.argv[2]
    this_folder = sys.argv[3]
    folders = sorted([x[0] for x in os.walk(this_folder)])[1:]
    equal = sys.argv[4]

    Xf = open(OUT_FILE_X,'w')
    Xwriter = csv.writer(Xf)
    Yf = open(OUT_FILE_Y,'w')
    Ywriter = csv.writer(Yf)

    y = 0
    
    if equal == 'True':
        amounts = []
        for folder in folders:
            amounts.append(len(glob.glob(folder + '/*.jpg')))
        minimum = min(amounts)
        print minimum
        for folder in folders:
            folder_to_csv_min(folder, Xwriter, Ywriter, y, minimum)
            y=y+1
    else:
        for folder in folders:
            folder_to_csv(folder, Xwriter, Ywriter, y)
            y=y+1




