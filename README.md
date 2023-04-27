# IntellEvent

IntellEvent is an accurate and robust machine learning (ML) based gait event detection algorithm for 3D motion capturing data which automatically detects initial contact (IC) and foot off (FO) events during overground walking using a certain markerset. The underlying model was trained utilising a retrospective clinical 3D gait analysis dataset of 1211 patients and 61 healthy controls leading to 5717 trials with at least four gait events per trial determined by force plates. 
IntellEvent is only trained on ground truth (= force plate) data which ensures the quality of the training events (objective).



### Data information
The underlying model was trained utilising a retrospective clinical 3D gait analysis dataset of 1211 patients and 61 healthy controls (male: 664, female: 608, age: 18 $\pm$ 14.6 years). All persons were examined by a clinician and categorized into one of five classes depending on the underlying pathology - malrotation deformities of the lower limbs (MD, n=730), club foot (CF, n=120), infantile cerebral palsy (ICP n=344), ICP with only drop foot characteristics (DF, n=17), and healthy controls (HC, n=61). Overall, the dataset contains 5717 trials (a trial is defined as one recording including several consecutive steps in one direction) with at least four gait events per trial determined by force plates.
3D gait data was recorded on a 12-meter walkway at self-selected walking speed using 12 infrared cameras (Vicon Motion Systems Ltd, Oxford Metrics, UK) and three strain-gauge force plates (Advanced Mechanical Technology Inc., Watertown, MA) with a sampling rate of 150 Hz and 1500 Hz, respectively. 
For capturing kinematic data of the lower extremities, the extended Cleveland Clinic marker set was applied in combination with the Vicon Plug-In-Gait model for the upper body. The ground force detection threshold was set to 20 N. 3D trajectories of the markers were filtered using the Woltring filtering routine integrated in the Vicon Nexus system with an MSE value of 15. 

# Requirements
When using the current models (.h5 files) for the Vicon Nexus pipeline:
* Python 3.7, Numpy 1.18.5, Tensorflow 2.3.0, Pandas 1.2.0, ezc3d (https://github.com/pyomeca/ezc3d), Flask

General requirements:
* Python 3, Tensorflow, Keras, Numpy, Pandas, Scipy   


# Contact
bernhard.dumphart@fhstp.ac.at



