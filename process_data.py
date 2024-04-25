import numpy as np
import re
from scipy.io import loadmat
import os

folder = '/Users/shiqi/Documents/PhD/Code/power_grid/share_V4/data/'
save_dir = 'data/'
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

        # Load the data
        mat_data = loadmat(folder + f)

        # Check the keys
        (f"ErrorType: {ErrorType}, param: {param}, Index: {Index}")
        

        mat_data['ErrorType'] = ErrorType
        mat_data['param'] = param
        mat_data['Index'] = Index

        print(mat_data.keys())

        # Save the data
        np.save(f'{save_dir}Case_{ErrorType}_{param}_{Index}.npy', mat_data)
        print(f'Data has been saved in {save_dir}Case_{ErrorType}_{param}_{Index}.npy')
