import pandas as pd
import os
import sys
json_path = os.path.join(sys.path[1], 'yolov5/runs/test/exp5/best_predictions.json')
save_path = '/home/jayce/Documents/results.csv'
df = pd.read_json (json_path)
df.to_csv (save_path, index = None) #r'Path where the new CSV file will be stored\New File Name.csv'