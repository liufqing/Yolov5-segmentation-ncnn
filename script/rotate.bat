@echo off

set CWD=D:\Projects\C_Cpp\Yolo
set RELEASE=%CWD%\x64\Release

set INPUT=%CWD%\autogen
set OUTPUT_FOLDER=%CWD%\autogen-output

set LOG=%CWD%\autogen-log

cd %CWD%

for %%i in (%INPUT%\*.jpg) do (
    mkdir %OUTPUT_FOLDER%\%%~ni
    
    echo Rotating all images in %%~ni folder
    %RELEASE%\yolov5-seg.exe --source autogen\%%~ni --output autogen-output\%%~ni --rotate --save >> %LOG%\%%~ni_rotate.txt

    echo ------------------------------------------------
)

echo Finish !
pause