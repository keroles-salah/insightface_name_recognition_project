@echo off
set /p person_name=Enter person name: 
set /p samples=Enter number of samples, example 80: 
python src\collect_faces.py --name "%person_name%" --samples %samples%
pause