import sys

def write_head(fs):
    header = "\\relative c' \n { \\time 4/4 \n "
    fs.write(header)

def write_content(fr,fs):
    i=0
    for line in fr:
        if int(line[0])==0:
            next_note = 'r'
        if int(line[0])==1:
            next_note = 'c'
        fs.write(next_note + ' ')
        i=i+1
        if i==4:
            fs.write('\n ')
            i=0
        

def write_bottom(fs):
    fs.write('\n}')


if __name__ == '__main__':
    THIS_FOLDER = sys.argv[1]

    fr = open(THIS_FOLDER + '/notes_as_numbers.txt','r')
    fs = open(THIS_FOLDER + '/notes_as_lilypond.txt','w')

    write_head(fs)
    write_content(fr,fs)
    write_bottom(fs)


