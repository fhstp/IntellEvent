# Requirements
win-64, Python 3.7, numpy==1.18.5, tensorflow==2.3.0, scikit-learn==1.0.2, pandas==1.2.0, requests==2.28.1, flask==2.2.2, Vicon Nexus API (https://docs.vicon.com/display/Nexus212/Set+up+Python+for+use+with+Nexus), ezc3d (https://github.com/pyomeca/ezc3d)

# Step-by-step Setup
1. Install a preferred python IDE
   1. For example [PyCharm](https://www.jetbrains.com/pycharm/), [Spyder](https://www.spyder-ide.org/), [Visual Studio Code](https://code.visualstudio.com/), etc.)
   2. For installation refer to the specific websites
2. Install [Anaconda](https://www.anaconda.com/) as your package manager
     1. For installation refer to the specific website
3. Create a new Project
   1. After the installation of your preferred __IDE__ and Anaconda create a new Project with your preferred name and location.
5. Create a new Anaconda environment
   1. Create a new Anaconda environment in your __IDE__ or in the __'Anaconda Navigator'__ using __Python Version 3.7__
   2. Download the __'requirements.txt'__ from this folder and copy it to your specific __project folder__
   3. In the __'Terminal'__ of your __IDE__ enter the following code: ```pip install -r requirements.txt```
5. The package __ezc3d__ and the __Vicon Nexus API__ need to be installed separately.
  1. Enter the following code into your __'Terminal'__ to install the __ezc3d__ package: ```conda install -c conda-forge ezc3d```
6. Install the __Vicon Nexus API__
   1. For further information follow the steps on the [Vicon Documentation](https://docs.vicon.com/display/Nexus212/Set+up+Python+for+use+with+Nexus)
   2. Open a __Windows Terminal (cmd)__ and activate your __Anaconda Environment__: ```conda activate IntellEvent```
   3. Locate where __Vicon Nexus__ is installed on your PC and change to this folder. Recreate the following steps to install the __Vicon Nexus API__ into your environment:
``` 
cd "C:\Program Files (x86)\Vicon\Nexus2.12\SDK\Win32\Python"
# install the api into the system python
./install_vicon_nexus_api.bat
# install the api into a specific python distribution
pip install ./viconnexusapi
```

