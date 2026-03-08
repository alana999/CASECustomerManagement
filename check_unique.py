import pandas as pd

try:
    df_base = pd.read_csv('customer_base.csv')
    df_beh = pd.read_csv('customer_behavior_assets.csv')
    
    print('Lifecycle:', df_base['lifecycle_stage'].unique())
    print('Occupation Type:', df_base['occupation_type'].unique())
    print('Asset Level:', df_beh['asset_level'].unique())
    print('Contact Result:', df_beh['contact_result'].unique())
    
    # 检查关联性
    merged = pd.merge(df_base, df_beh, on='customer_id')
    print('Merged count:', len(merged))
except Exception as e:
    print(e)
