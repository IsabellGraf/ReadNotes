clean:
	rm -rf PREPROCESSED/*

preprocess: gerade verkleinert find_staves find_notes


gerade:
	mkdir PREPROCESSED/STRAIGHT
	python SCRIPTS/auto_rotate_scanned_notes.py PREPROCESSED/STRAIGHT ORIGINAL/*.jpg

verkleinert:
	mkdir PREPROCESSED/SHRINKED
	bash SCRIPTS/shrink_jpg.sh PREPROCESSED/SHRINKED/ PREPROCESSED/STRAIGHT/*.jpg

find_staves: 
	mkdir PREPROCESSED/STAVES
	python SCRIPTS/find_staves.py PREPROCESSED/STAVES/ PREPROCESSED/SHRINKED/*.jpg

find_notes:
	mkdir PREPROCESSED/NOTES
	python SCRIPTS/find_notes.py PREPROCESSED/NOTES/ PREPROCESSED/STAVES/*.jpg



write_song: clean_song classify_layers write_lilypond write_pdf


clean_song:
	rm -rf SONGPDF

classify_layers:
	mkdir SONGPDF
	python SCRIPTS/use_classifier.py SONGPDF PREPROCESSED/NOTES 


write_lilypond:
	python SCRIPTS/write_lilypond.py SONGPDF

write_pdf:
	bash SCRIPTS/write_pdf.sh SONGPDF

