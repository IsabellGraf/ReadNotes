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
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/A3/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/A4/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/A5/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/C4/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/C5/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/C6/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/D4/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/D5/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/D6/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/E4/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/E5/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/E6/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/F3/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/F4/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/F5/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/F6/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/G3/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/G4/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/G5/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/H3/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/H4/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/H5/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/NO/*.jpg

delete_variants:
	rm -f NOTES_YES_NO/YES/A3/*_*
	rm -f NOTES_YES_NO/YES/A4/*_*
	rm -f NOTES_YES_NO/YES/A5/*_*
	rm -f NOTES_YES_NO/YES/C4/*_*
	rm -f NOTES_YES_NO/YES/C5/*_*
	rm -f NOTES_YES_NO/YES/C6/*_*
	rm -f NOTES_YES_NO/YES/D4/*_*
	rm -f NOTES_YES_NO/YES/D5/*_*
	rm -f NOTES_YES_NO/YES/D6/*_*
	rm -f NOTES_YES_NO/YES/E4/*_*
	rm -f NOTES_YES_NO/YES/E5/*_*
	rm -f NOTES_YES_NO/YES/E6/*_*
	rm -f NOTES_YES_NO/YES/F3/*_*
	rm -f NOTES_YES_NO/YES/F4/*_*
	rm -f NOTES_YES_NO/YES/F5/*_*
	rm -f NOTES_YES_NO/YES/F6/*_*
	rm -f NOTES_YES_NO/YES/G3/*_*
	rm -f NOTES_YES_NO/YES/G4/*_*
	rm -f NOTES_YES_NO/YES/G5/*_*
	rm -f NOTES_YES_NO/YES/H3/*_*
	rm -f NOTES_YES_NO/YES/H4/*_*
	rm -f NOTES_YES_NO/YES/H5/*_*
	rm -f NOTES_YES_NO/NO/*_*

classify_notes: classify_layers

notes_to_csv: 
	python SCRIPTS/notes_to_csv.py X_notes.csv Y_notes.csv NOTES_YES_NO/ False False

classify:
	python SCRIPTS/classify_notes.py NOTES_YES_NO equal=True

classify_layers:
	python SCRIPTS/classify_notes_multiple_layers.py NOTES_YES_NO/ equal=False


