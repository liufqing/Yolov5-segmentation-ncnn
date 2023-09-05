@echo off

set CWD=D:\Projects\C_Cpp\Yolo
set RELEASE=%CWD%\x64\Release

set INPUT=%CWD%\autogen
set OUTPUT_FOLDER=%CWD%\autogen-output

set LOG=%CWD%\autogen-log

cd %CWD%

for %%i in (%INPUT%\*.jpg) do (
    echo Compare angles %%~ni
    %RELEASE%\compareAngles.exe %OUTPUT_FOLDER%\%%~ni\rotate\angle.txt >> %LOG%\%%~ni_compareAngles.txt

    echo ------------------------------------------------
)

echo Finish !
pause
