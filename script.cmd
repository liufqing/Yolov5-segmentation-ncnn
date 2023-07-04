x64\Release\autoGenRotate autogen\cccdc_front.jpg -89 90 0.7 >> cccdc_front.txt
x64\Release\yoloncnn --source autogen\cccdc_front --output output-autogen\cccdc_front --rotate --save >> cccdc_front.txt
x64\Release\compareAngle D:\Projects\Random_Project\Yolo\output-autogen\cccdc_front\angle.txt >> cccdc_front.txt
pause