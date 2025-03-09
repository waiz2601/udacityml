import os 
import joblib
file_path="../final_project/poi_names.txt"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    print(f"Found file: {file_path}")
try:
    with open(file_path, "r") as f:
        lines=f.readlines()
    poi_count=0
    for line in lines:
        if line.startswith("(y)") or line.startswith("(n)"):
            poi_count+=1
    print("Number of POIs:", poi_count)
except Exception as e:
    print(f"Error loading file: {e}")            