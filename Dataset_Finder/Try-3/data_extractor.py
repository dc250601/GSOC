from runner import runner
from british import curzon
param = [25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500]
for parameter in param:
    print(f"entering runner(parameter = {parameter})...")
    runner(parameter)
    print(f"entering curzon(parameter = {parameter})...")
    curzon(parameter)
    print("curzon completed!")