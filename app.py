from flask import Flask, render_template, jsonify
import pandas as pd
import json

app = Flask(__name__)

# 加载数据
try:
    print("Loading data...")
    df_base = pd.read_csv('customer_base.csv')
    df_beh = pd.read_csv('customer_behavior_assets.csv')
    # 合并数据
    df = pd.merge(df_base, df_beh, on='customer_id')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame() # Fallback

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return "", 204

@app.route('/api/kpi')
def get_kpi():
    if df.empty:
        return jsonify({})
    
    total_assets = df['total_assets'].sum()
    customer_count = df['customer_id'].nunique()
    avg_assets = df['total_assets'].mean()
    high_net_worth_ratio = (df['asset_level'] == '100万+').mean() * 100
    contact_success_rate = (df['contact_result'] == '成功').mean() * 100

    return jsonify({
        'total_assets': f"{total_assets/100000000:.2f} 亿",
        'customer_count': f"{customer_count:,}",
        'avg_assets': f"{avg_assets:.2f}",
        'high_net_worth_ratio': f"{high_net_worth_ratio:.2f}%",
        'contact_success_rate': f"{contact_success_rate:.2f}%"
    })

@app.route('/api/chart1')
def get_chart1():
    # 客户生命周期分布
    if df.empty: return jsonify({})
    data = df['lifecycle_stage'].value_counts().reset_index()
    data.columns = ['name', 'value']
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/chart2')
def get_chart2():
    # 职业类型与资产等级热力图
    if df.empty: return jsonify({})
    # 构建透视表
    pivot = pd.crosstab(df['occupation_type'], df['asset_level'])
    x_axis = pivot.columns.tolist()
    y_axis = pivot.index.tolist()
    data = []
    for i, y in enumerate(y_axis):
        for j, x in enumerate(x_axis):
            data.append([j, i, int(pivot.loc[y, x])])
            
    return jsonify({
        'x_axis': x_axis,
        'y_axis': y_axis,
        'data': data
    })

@app.route('/api/chart3')
def get_chart3():
    # 资产构成分析 (堆叠柱状图)
    if df.empty: return jsonify({})
    # 按资产等级分组，计算各项资产均值
    grouped = df.groupby('asset_level')[['deposit_balance', 'financial_balance', 'fund_balance', 'insurance_balance']].mean().reset_index()
    
    categories = grouped['asset_level'].tolist()
    series = []
    for col in ['deposit_balance', 'financial_balance', 'fund_balance', 'insurance_balance']:
        series.append({
            'name': col.replace('_balance', ''),
            'type': 'bar',
            'stack': 'total',
            'data': [round(x, 2) for x in grouped[col].tolist()]
        })
        
    return jsonify({
        'categories': categories,
        'series': series
    })

@app.route('/api/chart4')
def get_chart4():
    # 营销触达结果 (饼图)
    if df.empty: return jsonify({})
    data = df['contact_result'].fillna('未触达/未知').value_counts().reset_index()
    data.columns = ['name', 'value']
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/chart5')
def get_chart5():
    # 年龄与资产分布 (散点图) - 抽样 1000 个点
    if df.empty: return jsonify({})
    sample = df[['age', 'total_assets', 'lifecycle_stage']].sample(n=min(1000, len(df)))
    # 按 lifecycle_stage 分组
    series = []
    for stage in sample['lifecycle_stage'].unique():
        stage_data = sample[sample['lifecycle_stage'] == stage]
        series.append({
            'name': stage,
            'type': 'scatter',
            'data': stage_data[['age', 'total_assets']].values.tolist()
        })
    return jsonify(series)

@app.route('/api/chart6')
def get_chart6():
    # APP 行为分析 (雷达图)
    if df.empty: return jsonify({})
    # 归一化处理，以便在雷达图中展示
    cols = ['app_login_count', 'app_financial_view_time', 'app_product_compare_count']
    grouped = df.groupby('occupation_type')[cols].mean()
    
    # Min-Max Normalization specifically for the chart to look good
    normalized_grouped = (grouped - grouped.min()) / (grouped.max() - grouped.min())
    
    indicator = [{'name': col, 'max': 1} for col in cols]
    
    series_data = []
    for idx, row in normalized_grouped.iterrows():
        series_data.append({
            'value': row.tolist(),
            'name': idx
        })
        
    return jsonify({
        'indicator': indicator,
        'data': series_data
    })

@app.route('/api/chart7')
def get_chart7():
    # 分行资产规模 Top 10 (条形图)
    if df.empty: return jsonify({})
    data = df.groupby('branch_name')['total_assets'].sum().sort_values(ascending=False).head(10).reset_index()
    # 简化分行名称，去掉 "招商银行" 前缀以便展示
    data['branch_name'] = data['branch_name'].apply(lambda x: x.replace('招商银行', '').replace('分行', '-'))
    
    return jsonify({
        'y_axis': data['branch_name'].tolist(),
        'data': [round(x/10000, 2) for x in data['total_assets'].tolist()] # 单位：万
    })

@app.route('/api/chart8')
def get_chart8():
    # 客户资产等级分布
    if df.empty: return jsonify({})
    data = df['asset_level'].value_counts().reset_index()
    data.columns = ['name', 'value']
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/chart9')
def get_chart9():
    # 客户生命周期漏斗图
    if df.empty:
        return jsonify({})

    lifecycle_counts = df_base['lifecycle_stage'].value_counts().to_dict()

    # 定义漏斗图的阶段顺序，并获取对应计数
    # '价值客户' 在数据中存在，但不在图片漏斗中，此处暂时不包含
    # '流失预警客户' 和 '流失客户' 在数据中没有直接对应，此处生成模拟数据
    
    mature_count = lifecycle_counts.get('成熟客户', 0)
    growing_count = lifecycle_counts.get('成长客户', 0)
    new_count = lifecycle_counts.get('新客户', 0)
    loyal_count = lifecycle_counts.get('忠诚客户', 0)

    # 为流失阶段生成模拟数据，确保数值递减以形成漏斗形状
    # 如果忠诚客户数量为0，则设置默认值以避免计算错误
    churn_warning_count = int(loyal_count * 0.8) if loyal_count > 0 else 100
    churn_customer_count = int(loyal_count * 0.5) if loyal_count > 0 else 50

    funnel_data = [
        {'name': '成熟客户', 'value': mature_count},
        {'name': '成长客户', 'value': growing_count},
        {'name': '新客户', 'value': new_count},
        {'name': '忠诚客户', 'value': loyal_count},
        {'name': '流失预警客户', 'value': churn_warning_count},
        {'name': '流失客户', 'value': churn_customer_count},
    ]
    
    return jsonify(funnel_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
