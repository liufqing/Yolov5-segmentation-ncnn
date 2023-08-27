@echo off

set RELEASE=D:\Projects\Random_Project\Yolo\x64\Release
set CWD=D:\Projects\Random_Project\Yolo

set INPUT=D:\Projects\Random_Project\Yolo\autogen
set OUTPUT_FOLDER=D:\Projects\Random_Project\Yolo\autogen-output

set LOG=D:\Projects\Random_Project\Yolo\autogen-log

cd %CWD%

for %%i in (%INPUT%\*.jpg) do (
    mkdir %OUTPUT_FOLDER%\%%~ni
    
    echo Rotating all images in %%~ni folder
    %RELEASE%\yoloncnn ^
    --source autogen\%%~ni ^
    --output autogen-output\%%~ni ^
    --rotate ^
    --save >> %LOG%\%%~ni_rotate.txt

    echo ------------------------------------------------
)

echo Finish !
pause