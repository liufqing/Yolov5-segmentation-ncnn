@echo off

set CWD=D:\Projects\C_Cpp\Yolo
set RELEASE=%CWD%\x64\Release

set OUTPUT_FOLDER=%CWD%\temp
set MODEL_PATH=%CWD%\models/yolov5s-seg-idcard-best-2.ncnn

set LOG=%CWD%\autogen-log

cd %CWD%

set /p INPUT=enter images folder : 

rem echo %INPUT%

for %%i in (%INPUT%\*.jpg) do (
   echo %%i
   rem  mkdir %OUTPUT_FOLDER%\%%~ni
    
    %RELEASE%\yolov5-seg.exe --source %%i --output %OUTPUT_FOLDER% --model %MODEL_PATH% --save

    echo ------------------------------------------------
)

echo Finish !
pause