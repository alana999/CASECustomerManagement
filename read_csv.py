import pandas as pd
import sys

# 设置 pandas 显示选项，以显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

files = ['customer_base.csv', 'customer_behavior_assets.csv']

for file in files:
    try:
        print(f"Reading {file}...")
        df = pd.read_csv(file)
        print(df.head(5))
        print("-" * 50)
    except FileNotFoundError:
        print(f"Error: {file} not found.")
    except Exception as e:
        print(f"Error reading {file}: {e}")
