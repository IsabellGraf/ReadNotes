TARGET_DIR=$1

for f in "${@:2}";
do
	base=$(basename $f)
	convert $f -resize 1000 $TARGET_DIR$base
done

