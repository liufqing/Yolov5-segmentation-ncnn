@echo off

set RELEASE=D:\Projects\Random_Project\Yolo\x64\Release
set CWD=D:\Projects\Random_Project\Yolo

set INPUT=D:\Projects\Random_Project\Yolo\autogen
set OUTPUT_FOLDER=D:\Projects\Random_Project\Yolo\autogen-output

set LOG=D:\Projects\Random_Project\Yolo\autogen-log

cd %CWD%

call script\gen.bat

call script\rotate.bat

call script\compareAngles.bat
