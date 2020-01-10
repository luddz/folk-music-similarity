# Annotation Distances With DTW

Uses the probability distribution for each token in the annotation to create a matrix of distributions. Use DTW to calculate a distance between these matrices. This is a modification of the code found [here](https://github.com/IraKorshunova/folk-rnn)

## How-to
Run in two steps

### 1. Create matrices
Use the python script `sample-rnn-fast.py` with a transcription string as an seed. This will save the matrix of distributions to a file.

#### Arguments

* `[metadata_path]` The path to the .pkl metadata-file
* `[--seed]` A string containing a transcription
* `[--transcription_id]` An integer identifying this transcription (Used as name for the outputed file)

#### Sample input

`python2 sample-rnn-fast.py config5-wrepeats-20160112-222521.pkl --seed "M:4/4 K:Cdor c 3 d c 2 B G | G F F 2 G B c d | c 3 d c B G A | B G G F G 2 f e | d 2 c d c B G A | B G F G B F G F | G B c d c d e f | g b f d e 2 e f | d B B 2 B 2 d c | B G G 2 B G F B | d B B 2 c d e f | g 2 f d g f d c | d g g 2 f d B c | d B B 2 B G B c | d f d c B 2 d B | c B B G F B G B | d 2 c d d 2 f d | g e c d e 2 f g | f d d B c 2 d B | d c c B G B B c | d 2 c d d 2 f d | c d c B c 2 d f | g b b 2 g a b d' | c' d' b g f f f f |" --transcription_id 1`


This will generate a file called `matrix_1.txt`


### 2. Calculate the DTW distance between two matrices
Use the python script `dtw_distance.py` to calculate the DTW distance between two matrices.

#### Arguments

* `[--first_id]` The id of the first matrix .txt file
* `[--second_id]` The id of the first matrix .txt file

#### Sample input
`python2 dtw_distance.py --first_id 1 --second_id 2`

This will print the DTW distance of the matrices in files `matrix_1.txt` and `matrix_2.txt`
