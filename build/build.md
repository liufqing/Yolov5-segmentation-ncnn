# Setup
Setup for windows, using visual studio 2022
## Prerequisites
### 1. Install Visual Studio 2022 [download visual studio community](https://visualstudio.microsoft.com/vs/community/)

### 2. Install Cmake [download cmake](https://cmake.org/download/)
1. First, Get the lastest cmake pre-compiled binary on this [download page](https://cmake.org/download/).
2. If download the Windows MSI installer. During installation, remember to select the option "Add Cmake to the system Path (for all users or current user)". Then you can skip step 3.
![screenshot1](./screenshots/Screenshot1.png)
3. If that is not selected during installation or if you download from a zip file, you need to manually add the install directory (e.g. C:\Program Files\CMake\bin) to the system variable :
![screenshot2](./screenshots/Screenshot2.png)
![screenshot3](./screenshots/Screenshot3.png)
![screenshot4](./screenshots/Screenshot4.png)
![screenshot5](./screenshots/Screenshot5.png)

### 3. Install OpenCV framework [OpenCV Github repository](https://github.com/opencv/opencv)
1. Download and install the [latest release](https://github.com/opencv/opencv/releases/). For me, it is the "opencv-4.7.0-windows.exe"
2. Add the bin folder to the Environment Variables path : *\opencv\build\x64\vc16\bin*
![screenshot6](./screenshots/Screenshot6.png)

### 4. Install The Tencent's NCNN framework [NCNN Github repository](https://github.com/Tencent/ncnn)
To build for Visual Studio, refer to [this](https://github.com/Tencent/ncnn/wiki/build-for-VisualStudio.en#compile-and-install-ncnn-library-and-model-conversion-tool) tutorial

## Setup Visual Studio Project
1. Create a New Visual Studio project C++ console
2. Set the platform target to x64
3. Add Directories by going to *Project>Properties>Configuration Properties>VC++ Directories*. Add opencv and ncnn include folder to *Include directories* and add library folder to *Library Directories*
![screenshot7](./screenshots/Screenshot7.png)
![screenshot8](./screenshots/Screenshot8.png)
![screenshot9](./screenshots/Screenshot9.png)
4. Add Dependencies by going to *Project>Properties>Configuration Properties>Linker>Input*. Add opencv and ncnn *.lib* to *Additiona Dependencies* (d for debug, without d for release )
![screenshot10](./screenshots/Screenshot10.png)
![screenshot11](./screenshots/Screenshot11.png)