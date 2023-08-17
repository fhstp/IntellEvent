# IntellEvent

IntellEvent is an accurate and robust machine learning (ML) based gait event detection algorithm for 3D motion capturing data which automatically detects initial contact (IC) and foot off (FO) events during overground walking using a certain markerset. The underlying model was trained utilising a retrospective clinical 3D gait analysis dataset of 1211 patients and 61 healthy controls leading to 5717 trials with at least four gait events per trial determined by force plates. 
IntellEvent is only trained on ground truth (= force plate) data which ensures the quality of the training events (objective). For further information visite the publication website [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0288555).



# Requirements
When using the current models (.h5 files) for the Vicon Nexus pipeline:
* Python 3.7, Numpy 1.18.5, Tensorflow 2.3.0, Pandas 1.2.0, ezc3d (https://github.com/pyomeca/ezc3d), Flask

General requirements:
* Python 3, Tensorflow, Keras, Numpy, Pandas, Scipy   


# Contact
bernhard.dumphart@fhstp.ac.at


## License
Attribution-NonCommercial-ShareAlike 4.0 Internationa
