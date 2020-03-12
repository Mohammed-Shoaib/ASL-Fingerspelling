#!/bin/bash

base="fingerspelling-asl"
data="american-sign-language"
letters="ABCDEFGHIKLMNOPQRSTUVWXY"	# excluding J & Z

cd .. # change to parent directory

# create directories if they do not exist
mkdir -p data
for (( i = 0; i < ${#letters}; i++ )); do	# loop over string
    l="${letters:$i:1}"
    mkdir -p data/$l/
done

# download & extract dataset if not present
if [ ! -f "$data.zip" ]; then
	echo "Downloading the kaggle dataset..."
	kaggle datasets download shoaib98libra/$data
fi
if [ ! -d "$base" ]; then
	echo "Extracting the kaggle dataset..."
	unzip "$data.zip"
fi

echo "Copying the dataset into one directory..."
subjects=$(ls -d $base/subject* | wc -l)
for (( i = 1; i <= $subjects; i++ )); do		# loop over the subjects
	for (( j = 0; j < ${#letters}; j++ )); do	# loop over string
	    l="${letters:$j:1}"
	    cp --backup=numbered $base/subject-$i/$l/* data/$l/
	done
	echo "Finished subject $i of $subjects."
done