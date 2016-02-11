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
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/A/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/a/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/aa/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/C/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/c/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/cc/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/D/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/d/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/dd/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/E/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/e/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/ee/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/F/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/f/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/ff/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/fff/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/G/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/g/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/gg/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/H/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/h/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/YES/hh/*.jpg
	python SCRIPTS/create_variants.py NOTES_YES_NO/NO/*.jpg

delete_variants:
	rm -f NOTES_YES_NO/YES/A/*_*
	rm -f NOTES_YES_NO/YES/a/*_*
	rm -f NOTES_YES_NO/YES/aa/*_*
	rm -f NOTES_YES_NO/YES/C/*_*
	rm -f NOTES_YES_NO/YES/c/*_*
	rm -f NOTES_YES_NO/YES/cc/*_*
	rm -f NOTES_YES_NO/YES/D/*_*
	rm -f NOTES_YES_NO/YES/d/*_*
	rm -f NOTES_YES_NO/YES/dd/*_*
	rm -f NOTES_YES_NO/YES/E/*_*
	rm -f NOTES_YES_NO/YES/e/*_*
	rm -f NOTES_YES_NO/YES/ee/*_*
	rm -f NOTES_YES_NO/YES/F/*_*
	rm -f NOTES_YES_NO/YES/f/*_*
	rm -f NOTES_YES_NO/YES/ff/*_*
	rm -f NOTES_YES_NO/YES/fff/*_*
	rm -f NOTES_YES_NO/YES/G/*_*
	rm -f NOTES_YES_NO/YES/g/*_*
	rm -f NOTES_YES_NO/YES/gg/*_*
	rm -f NOTES_YES_NO/YES/H/*_*
	rm -f NOTES_YES_NO/YES/h/*_*
	rm -f NOTES_YES_NO/YES/hh*_*
	rm -f NOTES_YES_NO/NO/*_*

classify_notes: classify_layers

notes_to_csv: 
	python SCRIPTS/notes_to_csv.py X_notes.csv Y_notes.csv NOTES_YES_NO/ False False

classify:
	python SCRIPTS/classify_notes.py NOTES_YES_NO equal=True

classify_layers:
	python SCRIPTS/classify_notes_multiple_layers.py NOTES_YES_NO/ equal=False


