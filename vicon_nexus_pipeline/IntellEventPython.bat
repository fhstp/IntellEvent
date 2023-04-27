@echo off
:: This is the .bat file which you have to adapt
:: for the Vicon Nexus (2.14.0) pipeline to your virtual environment!
:: add a new 'Run Python Operation' into your Vicon Nexus processing
:: pipeline and click on 'Show Advanced'. There you can find 'Environment activation'
:: where a 'NexusLocalPython.bat' file should be located. Here you should add your
:: own .bat file for the activation. 
:: (NOTE: this works ONLY for the Vicon Nexus Version 2.14.0 and above!)
:: (NOTE: get in contact with us for the use on older Vicon Nexus Versions!)


:: the line 'call activate IntellEventNew' represents a conda environment with
:: the name 'IntellEventNew' -> this will activate your conda environment.
call activate IntellEventNew
:: then you have to set the path to your conda environment you want to use
:: as well as to the 'scripts' folder in this conda environment
set PATH=C:\Users\admin\anaconda3\envs\IntellEventNew;C:\Users\admin\anaconda3\envs\IntellEventNew\Scripts;%PATH%
python --version
