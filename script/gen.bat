@echo off

set RELEASE=D:\Projects\Random_Project\Yolo\x64\Release
set CWD=D:\Projects\Random_Project\Yolo

set INPUT=D:\Projects\Random_Project\Yolo\autogen
set OUTPUT_FOLDER=D:\Projects\Random_Project\Yolo\autogen-output

set LOG=D:\Projects\Random_Project\Yolo\autogen-log

cd %CWD%

for %%i in (%INPUT%\*.jpg) do (
    echo Generating image from %%i
    %RELEASE%\autoGenRotate.exe %%i -150 180 30 0.5 >> %LOG%\%%~ni_gen.txt
    
    echo ------------------------------------------------
)

echo Finish !
pause
