@echo off

set CWD=D:\Projects\C_Cpp\Yolo
set RELEASE=%CWD%\x64\Release

set INPUT=%CWD%\autogen
set OUTPUT_FOLDER=%CWD%\autogen-output
set MODEL_PATH=%CWD%\models/yolov5s-seg-idcard-best-2.ncnn

set LOG=%CWD%\autogen-log

cd %CWD%
 
for %%i in (%INPUT%\*.jpg) do (
    rem mkdir %OUTPUT_FOLDER%\%%~ni
    
    echo Rotating all images in %%~ni folder
    %RELEASE%\yolov5-seg.exe --source %INPUT%\%%~ni --output %OUTPUT_FOLDER%\%%~ni --model %MODEL_PATH% --rotate --save >> %LOG%\%%~ni_rotate.txt

    echo ------------------------------------------------
)

echo Finish !
pause