import numpy as np
import re
from scipy.io import loadmat
import os

folder = '/Users/shiqi/Documents/PhD/Code/Project3-power-grid/share_V4/data'
save_dir = '../data/data_selected'
files = os.listdir(folder)
files = [f for f in files if f.endswith('.mat')]

# Extract the case number from the file name
for f in files:
    print(f)
    match = re.match(r'Case_(\d+)_(\d+)_(\d+).mat', f)
    if match:
        ErrorType = int(match.group(1))
        param = int(match.group(2))
        Index = int(match.group(3))

        # Full path
        full_path = os.path.join(folder, f)
        
        # Check if the file exists
        if not os.path.isfile(full_path):
            print(f"File not found: {full_path}")
            continue

        # Load the data
        mat_data = loadmat(full_path)

        # Check the keys
        print(f"ErrorType: {ErrorType}, param: {param}, Index: {Index}")

        mat_data['ErrorType'] = ErrorType
        mat_data['param'] = param
        mat_data['Index'] = Index

        print(mat_data.keys())

        # Save the data
        save_path = os.path.join(save_dir, f'Case_{ErrorType}_{param}_{Index}.npy')
        np.save(save_path, mat_data)
        print(f'Data has been saved in {save_path}')
