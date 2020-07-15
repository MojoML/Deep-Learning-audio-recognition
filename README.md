Audio Classification challenge

Dataset
Speech classification
In this challenge the task is to learn to recognize which of several English words
is pronounced in an audio recording. This is a multiclass classification task.
Data files
The dataset is available for download on https://surfdrive.surf.nl/files/index.php/s/A91xgk7B5kXNvfJ. 

It contains the following files:

• wav.tgz: a compressed directory with all the recordings (training and test
data) in the form of wav files.

• feat.npy: an array with Mel-frequency cepstral coefficients extracted from
eachwavfile. The features at indexiin this array were extracted from
the wav file at index i of the array in the file path.npy.

• path.npy: an array with the order of wav files in the feat.npy array.

• train.csv: this file contains two columns: path with the filename of the
recording andwordwith word which was pronounced in the recording.
This is the training portion of the data.

• test.csv: This is the testing portion of the data, and it has the same
format as the file train.csv except that the column word is absent.
You can load the filesnpyusing the functionnumpy.load, and the CSV files
using the csv module or the pandas.read_csv function.

Evaluation metric
The evaluation metric for this task is classification accuracy (the proportion of
correct predictions).
Method
There are three important restrictions on the method used:

•the method should be fully automatic, that is, by re-running your code it
should be possible to re-create your prediction file;

•every software component used should be open-source and possible to
install locally. This means that you cannot use propritary closed-source
speech recognition software, or access a web service to carry out any data
processing;

•the method should not use any external dataset which overlaps with the
provided data. If you wish to make use of external data in your solution,
ask the instructor via the course forum to confirm that this data is allowed.
Some hints:

•You can use the provided MFCC features for the spoken utterances, or
you can extract your own features from the wav files.

•Use part of the provided training data as a validation set. 
