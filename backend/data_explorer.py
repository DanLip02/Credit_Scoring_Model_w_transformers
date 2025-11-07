import pandas as pd
import kagglehub
import os


def load_german_credit_risk():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Поднимаемся из backend в корень
    file_path = os.path.join(project_root, 'data', 'german_credit_data.csv')

    print(f"Searching file by path : {file_path}")
    df = pd.read_csv(file_path)

    return df

if __name__ == '__main__':
    pass