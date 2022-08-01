from runner1 import runner
from british import curzon
print("Making data for test 1, 50,200")
t = [1e-3, 3.2e-3, 1e-2, 3.2e-2, 1e-1, 3.2e-1]
for threshold in t:
    runner(threshold=threshold)
    curzon(threshold)
    print("curzon completed!")
