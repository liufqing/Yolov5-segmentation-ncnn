@echo off

set RELEASE=D:\Projects\Random_Project\Yolo\x64\Release
set CWD=D:\Projects\Random_Project\Yolo

set INPUT=D:\Projects\Random_Project\Yolo\autogen
set OUTPUT_FOLDER=D:\Projects\Random_Project\Yolo\autogen-output

set LOG=D:\Projects\Random_Project\Yolo\autogen-log

cd %CWD%

for %%i in (%INPUT%\*.jpg) do (
    echo Compare angles %%~ni
    %RELEASE%\compareAngles.exe %OUTPUT_FOLDER%\%%~ni\rotate\angle.txt >> %LOG%\%%~ni_compareAngles.txt

    echo ------------------------------------------------
)

echo Finish !
pause
