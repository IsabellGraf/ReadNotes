clean:
	rm -rf PREPROCESSED/*

preprocess: gerade verkleinert find_staves find_notes


gerade:
	mkdir PREPROCESSED/STRAIGHT
	python SCRIPTS/auto_rotate_scanned_notes.py PREPROCESSED/STRAIGHT ORIGINALE/*.jpg

verkleinert:
	mkdir PREPROCESSED/SHRINKED
	bash SCRIPTS/shrink_jpg.sh PREPROCESSED/SHRINKED/ PREPROCESSED/STRAIGHT/*.jpg

find_staves: 
	mkdir PREPROCESSED/STAVES
	python SCRIPTS/find_staves.py PREPROCESSED/STAVES/ PREPROCESSED/SHRINKED/*.jpg

find_notes:
	mkdir PREPROCESSED/NOTES
	python SCRIPTS/find_notes.py PREPROCESSED/NOTES/ PREPROCESSED/STAVES/*.jpg

create_variants:
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/NO/*.jpg

delete_variants:
	rm -f NOTES_YES_NO/YES/*_*
	rm -f NOTES_YES_NO/NO/*_*

classify_notes: classify

notes_to_csv: 
	python SCRIPTS/notes_to_csv.py X_notes.csv Y_notes.csv NOTES_YES_NO/ False False

classify:
	python SCRIPTS/classify_notes_new.py NOTES_YES_NO

classify_layers:
	python SCRIPTS/classify_notes_multiple_layers.py NOTES_YES_NO


