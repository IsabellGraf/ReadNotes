import numpy as np
import csv
import glob
import sys
import os
from PIL import Image, ImageDraw

def add_artifact(image_file):
    im = Image.open(image_file)

    x, y = im.size
    eX, eY = 20, 20  # Size of Bounding Box for ellipse

    bbox = (x / 2 - eX / 2, y / 2 - eY / 2, x / 2 + eX / 2, y / 2 + eY / 2)
    draw = ImageDraw.Draw(im)
    draw.ellipse(bbox, fill="black")
    del draw
    im_array = np.asarray(im)
    return im_array
    #im.save("output.png")
    # im.show()


def folder_to_csv_min(folder, Xwriter, Ywriter, y, minimum, with_artifact):
    for note_file in sorted(glob.glob(folder + '/*.jpg'))[:minimum]:
        if with_artifact:
            note = add_artifact(note_file)
        else:
            note = np.asarray(Image.open(note_file))
        to_save = np.reshape(note,(1,note.shape[0]*note.shape[1]))
        Xwriter.writerows(to_save)
        Y = np.reshape(np.array(y),(1,1))   
        Ywriter.writerows(Y)

def folder_to_csv(folder, Xwriter, Ywriter, y, with_artifact):
    for note_file in sorted(glob.glob(folder + '/*.jpg')):
        if with_artifact:
            note = add_artifact(note_file)
        else:
            note = np.asarray(Image.open(note_file))
        to_save = np.reshape(note,(1,note.shape[0]*note.shape[1]))
        Xwriter.writerows(to_save)
        Y = np.reshape(np.array(y),(1,1))   
        Ywriter.writerows(Y)

            
if __name__ == '__main__':
    OUT_FILE_X = sys.argv[1]
    OUT_FILE_Y = sys.argv[2]
    this_folder = sys.argv[3]
    folders = sorted([x[0] for x in os.walk(this_folder)])[1:]
    equal = sys.argv[4]     # True means that output variants have the same amount
    with_artifact = sys.argv[5]

    Xf = open(OUT_FILE_X,'w')
    Xwriter = csv.writer(Xf)
    Yf = open(OUT_FILE_Y,'w')
    Ywriter = csv.writer(Yf)

    print equal
    print with_artifact
    y = 0
    
    if equal == 'True':    # True means that output variants have the same amount
        amounts = []
        for folder in folders:
            amounts.append(len(glob.glob(folder + '/*.jpg')))
        minimum = min(amounts)
        print minimum
        for folder in folders:
            if with_artifact == 'True' and y==1:
                folder_to_csv_min(folder, Xwriter, Ywriter, y, minimum, True)
            else:
                folder_to_csv_min(folder, Xwriter, Ywriter, y, minimum, False)
            y=y+1
    else:
        for folder in folders:
            if with_artifact == 'True' and y==1:
                folder_to_csv(folder, Xwriter, Ywriter, y, True)
            else:
                folder_to_csv(folder, Xwriter, Ywriter, y, False)
            y=y+1




