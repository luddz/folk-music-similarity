#!/bin/bash
cd test/

#Convert songs from json2abc
echo "json2abc"
time python3 ../json2abc.py > ./songs.abc

mkdir midis

#Convert abc to midi
echo "abc2midi"
time abc2midi ./songs.abc -quiet -silent
mv *.mid midis/

#Feature extract midi
echo "midi2feature"
time java -Xmx6g -jar ../jMIR_3_1_developer/jSymbolic2/dist/jSymbolic2.jar -configrun ../jsymbolic_configurations.txt ./midis ./feature_vals.xml ./feature_desc.xml

#Train PCA
#echo "midi2feature"