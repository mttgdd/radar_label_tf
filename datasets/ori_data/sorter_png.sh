
for f in *.zip
do
	echo $f
	prefix="${f%%.*}"
	echo $prefix
	if [ ! -d "$prefix" ]; then
		unzip -qq $f
		mv qui/ "$prefix"/
		cd $prefix
		ls -l | wc -l
		rename 's/[ (][^)]*[)]//g' *.png
		rename 's/^image-0-//' *.png
		echo * | grep -oP '\d+(?=\.)' > "$prefix".txt
		find *.png -maxdepth 0 -exec mv {} "$prefix"_{} \;
		cd ../ 
	fi
done
