
for f in *.zip
do
	echo $f
	prefix="${f%%.*}"
	echo $prefix
	if [ ! -d "$prefix" ]; then
		unzip -q $f
		mv qui/ "$prefix"/
		cd $prefix
		ls -l | wc -l
        rename  's/\.scv/\.csv/' *.scv
		echo * | grep -oP '\d+(?=\.)' > "$prefix".txt
		find *.csv -maxdepth 0 -exec mv {} "$prefix"_{} \;
		cd ../ 
	fi
done
