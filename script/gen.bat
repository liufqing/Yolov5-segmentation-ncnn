@echo off

set RELEASE=D:\Projects\Random_Project\Yolo\x64\Release
set CWD=D:\Projects\Random_Project\Yolo

set INPUT=D:\Projects\Random_Project\Yolo\autogen
set OUTPUT_FOLDER=D:\Projects\Random_Project\Yolo\autogen-output

set LOG=D:\Projects\Random_Project\Yolo\autogen-log

cd %CWD%

for %%i in (%INPUT%\*.jpg) do (
    echo Generating image from %%i
    %RELEASE%\autoGenRotate.exe %%i -175 180 5 0.6 >> %LOG%\%%~ni_gen.txt
    
    echo ------------------------------------------------
)

echo Finish !
pause
