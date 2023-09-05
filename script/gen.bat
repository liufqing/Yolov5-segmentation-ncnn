@echo off

set CWD=D:\Projects\C_Cpp\Yolo
set RELEASE=%CWD%\x64\Release

set INPUT=%CWD%\autogen
set OUTPUT_FOLDER=%CWD%\autogen-output

set LOG=%CWD%\autogen-log

cd %CWD%

for %%i in (%INPUT%\*.jpg) do (
    echo Generating image from %%i
    %RELEASE%\autoGenRotate.exe %%i -175 180 5 >> %LOG%\%%~ni_gen.txt
    
    echo ------------------------------------------------
)

echo Finish !
pause
