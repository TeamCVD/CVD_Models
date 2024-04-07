import pandas as pd

def load_data(file_path):
    print('--------------------Loading data --------------------')
    return pd.read_csv(file_path)

