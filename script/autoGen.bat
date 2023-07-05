@echo off

set RELEASE=D:\Projects\Random_Project\Yolo\x64\Release
set CWD=D:\Projects\Random_Project\Yolo
cd %CWD%

mkdir results

set /p INPUT=input : 

%RELEASE%\autoGenRotate autogen\%INPUT%.jpg -45 45 0.7

mkdir output-autogen\%INPUT%

%RELEASE%\yoloncnn ^
    --source autogen\%INPUT% ^
    --output output-autogen\%INPUT% ^
    --rotate ^
    --save

%RELEASE%\compareAngles D:\Projects\Random_Project\Yolo\output-autogen\%INPUT%\rotate\angle.txt

pause