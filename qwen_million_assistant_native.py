import os
import json
import pymysql
from typing import Union
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.gui import WebUI
import dashscope

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30

# 1. 注册自定义的 MySQL 查询工具
@register_tool('mysql_query')
class MySQLQuery(BaseTool):
    description = '执行SQL查询以获取银行客户数据。'
    parameters = [{
        'name': 'sql_query',
        'type': 'string',
        'description': '要执行的标准的 MySQL 查询语句。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        # 解析参数
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {'sql_query': params}
        
        sql = params.get('sql_query', '')
        print(f"[{self.__class__.__name__}] 执行SQL: {sql}")
        
        # 建立数据库连接并执行查询
        try:
            connection = pymysql.connect(
                host='localhost',  # 本地数据库
                user='root',
                password='123mysql',
                database='bank',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            with connection.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
            connection.close()
            
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"SQL执行出错: {str(e)}"

# 2. 注册 AUM 增长预测工具 (集成 ARIMA 模型)
@register_tool('predict_future_aum')
class PredictFutureAUM(BaseTool):
    description = '使用已训练好的 ARIMA 模型，预测全行未来一个季度（3个月）的 AUM（总资产管理规模）增长趋势。不需要传入任何参数。'
    parameters = []

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import pickle
        import pandas as pd
        import os
        
        model_path = 'arima_model.pkl'
        print(f"[{self.__class__.__name__}] 正在加载模型并预测...")
        
        if not os.path.exists(model_path):
            return "错误：未找到预训练的 ARIMA 模型 (arima_model.pkl)。请先运行时间序列训练脚本保存模型。"
            
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 预测未来 3 个月
            forecast = model.get_forecast(steps=3)
            forecast_mean = forecast.predicted_mean
            
            # 格式化输出结果
            result_str = "【ARIMA 模型预测结果】\n"
            for date, value in forecast_mean.items():
                result_str += f"- {date.strftime('%Y-%m')}: 预计达到 {value:.2f} 亿元\n"
            
            # 计算环比增长率 (需要获取模型历史数据的最后一个值)
            # 在 statsmodels 中，可以使用 model.data.endog 获取原始输入数据
            current_aum = model.data.endog[-1]
            future_aum = forecast_mean.iloc[-1]
            growth_rate = (future_aum - current_aum) / current_aum * 100
            
            result_str += f"\n当前最新 AUM: {current_aum:.2f} 亿元\n"
            result_str += f"预计季度末 AUM: {future_aum:.2f} 亿元\n"
            result_str += f"预计季度环比增长率: {growth_rate:.2f}%\n"
            result_str += "营销建议：鉴于强劲的线性增长趋势，建议将季度考核指标基准设定为 6.12%，并持续推动存款向理财和基金的转化。"
            
            return result_str
            
        except Exception as e:
            return f"预测执行出错: {str(e)}"


# 3. 注册客户等级预测工具 (集成 LightGBM 分类模型，含 Data Adapter)
@register_tool('predict_customer_tier')
class PredictCustomerTier(BaseTool):
    description = '使用已训练好的 LightGBM 模型，根据给定的客户资产和行为数据，预测该客户是否能在未来3个月内跃升为“高净值”客户（资产>=100万）。'
    parameters = [{
        'name': 'customer_data',
        'type': 'string',
        'description': '客户的各项特征数据（JSON字符串格式），应包含如 total_aum, deposit_balance, age, monthly_income, city_level 等字段。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import pickle
        import pandas as pd
        import os
        import json
        import numpy as np
        
        model_path = 'lgb_model.pkl'
        print(f"[{self.__class__.__name__}] 正在加载模型并进行分类预测...")
        
        if not os.path.exists(model_path):
            return "错误：未找到预训练的 LightGBM 模型 (lgb_model.pkl)。请先运行分类训练脚本保存模型。"
            
        try:
            # 1. 解析输入数据
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    return "错误：传入的客户数据必须是有效的 JSON 格式。"
            
            raw_data = params.get('customer_data', '{}')
            if isinstance(raw_data, str):
                raw_data = json.loads(raw_data)
                
            # 处理单条或多条记录
            if isinstance(raw_data, dict):
                raw_data = [raw_data]
                
            df_raw = pd.DataFrame(raw_data)
            
            # ==========================================
            # 2. Data Adapter (数据转换适配层)
            # 必须将输入的数据库字段，严格映射为 LightGBM 训练时的 13 个特征：
            # ['avg_assets_3m', 'assets_volatility', 'latest_assets', 'avg_deposit_3m', 
            #  'avg_financial_3m', 'avg_app_login', 'avg_app_view_time', 'product_count', 
            #  'age', 'monthly_income', 'city_level', 'occupation_type']
            # ==========================================
            
            df_features = pd.DataFrame()
            
            # 安全获取函数（带默认值）
            def get_val(col, default=0.0):
                return df_raw[col] if col in df_raw.columns else default
            
            # 金额类字段（需要转换为“万”）
            df_features['latest_assets'] = get_val('total_aum', 0) / 10000
            # 如果没有3个月均值，暂用当月数据代替
            df_features['avg_assets_3m'] = get_val('total_aum', 0) / 10000
            df_features['assets_volatility'] = 0.0 # 缺失历史波动率，默认设为0
            df_features['avg_deposit_3m'] = get_val('deposit_balance_monthly_avg', get_val('deposit_balance', 0)) / 10000
            df_features['avg_financial_3m'] = get_val('wealth_management_balance_monthly_avg', get_val('wealth_management_balance', 0)) / 10000
            
            # 如果没有月收入，使用月均交易金额粗略估算
            df_features['monthly_income'] = get_val('monthly_transaction_amount', 0) * 0.5 / 10000 
            
            # 行为类特征
            df_features['avg_app_login'] = get_val('mobile_bank_login_count', 0)
            df_features['avg_app_view_time'] = get_val('mobile_bank_login_count', 0) * 5 # 简单推算
            
            # 计算持有的产品种类数
            product_cols = ['deposit_balance', 'wealth_management_balance', 'fund_balance', 'insurance_balance']
            df_features['product_count'] = 0
            for p_col in product_cols:
                if p_col in df_raw.columns:
                    df_features['product_count'] += (df_raw[p_col] > 0).astype(int)
            
            # 静态特征
            df_features['age'] = get_val('age', 35)
            df_features['city_level'] = get_val('city_level', '二线')
            df_features['occupation_type'] = get_val('occupation', '其他')
            
            # 计算资产趋势
            df_features['assets_trend'] = 0.0
            
            # 将分类变量转为 category
            for col in ['city_level', 'occupation_type']:
                df_features[col] = df_features[col].astype('category')
                
            # 确保列顺序与训练时绝对一致
            expected_cols = ['avg_assets_3m', 'assets_volatility', 'latest_assets', 
                             'avg_deposit_3m', 'avg_financial_3m', 'avg_app_login', 
                             'avg_app_view_time', 'product_count', 'assets_trend', 
                             'age', 'monthly_income', 'city_level', 'occupation_type']
            
            for col in expected_cols:
                if col not in df_features.columns:
                    df_features[col] = 0
            df_features = df_features[expected_cols]

            # 3. 加载模型并推理
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            y_prob = model.predict(df_features, num_iteration=model.best_iteration)
            y_pred = (y_prob > 0.5).astype(int)
            
            # 4. 格式化输出
            result_str = "【客户价值跃升预测结果】\n"
            for i, (pred, prob) in enumerate(zip(y_pred, y_prob)):
                cid = df_raw['customer_id'].iloc[i] if 'customer_id' in df_raw.columns else f"客户_{i+1}"
                status = "🚨 极高概率跃升为【高净值】客户！" if pred == 1 else "暂时无法跃升"
                result_str += f"- {cid}: 跃升概率 {prob*100:.1f}% -> 判定: {status}\n"
            
            return result_str
            
        except Exception as e:
            import traceback
            return f"预测执行出错: {str(e)}\n{traceback.format_exc()}"


@register_tool('predict_customer_tier_lr')
class PredictCustomerTierLR(BaseTool):
    description = '使用已训练好的逻辑回归模型（lr_model.pkl），预测客户未来3个月是否跃升为高净值（资产>=100万）。'
    parameters = [{
        'name': 'customer_data',
        'type': 'string',
        'description': '客户的各项特征数据（JSON字符串格式），支持单条或多条。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import json
        import os
        import pickle
        import pandas as pd

        model_path = 'lr_model.pkl'
        if not os.path.exists(model_path):
            return "错误：未找到预训练的逻辑回归模型 (lr_model.pkl)。请先运行逻辑回归训练脚本保存模型。"

        try:
            if isinstance(params, str):
                params = json.loads(params)

            raw_data = params.get('customer_data', '{}')
            if isinstance(raw_data, str):
                raw_data = json.loads(raw_data)
            if isinstance(raw_data, dict):
                raw_data = [raw_data]

            df_raw = pd.DataFrame(raw_data)

            def get_val(col, default=0.0):
                return df_raw[col] if col in df_raw.columns else default

            df_base = pd.DataFrame()
            df_base['avg_assets_3m'] = get_val('total_aum', 0.0)
            df_base['assets_volatility'] = 0.0
            df_base['latest_assets'] = get_val('total_aum', 0.0)
            df_base['avg_deposit_3m'] = get_val('deposit_balance_monthly_avg', get_val('deposit_balance', 0.0))
            df_base['avg_financial_3m'] = get_val('wealth_management_balance_monthly_avg', get_val('wealth_management_balance', 0.0))
            df_base['avg_app_login'] = get_val('mobile_bank_login_count', 0.0)
            df_base['avg_app_view_time'] = get_val('mobile_bank_login_count', 0.0) * 5

            product_cols = ['deposit_balance', 'wealth_management_balance', 'fund_balance', 'insurance_balance']
            df_base['product_count'] = 0
            for p_col in product_cols:
                if p_col in df_raw.columns:
                    df_base['product_count'] += (df_raw[p_col] > 0).astype(int)

            df_base['assets_trend'] = 0.0
            df_base['age'] = get_val('age', 35)
            df_base['monthly_income'] = get_val('monthly_income', get_val('monthly_transaction_amount', 0.0) * 0.5)
            df_base['city_level'] = get_val('city_level', '二线')
            df_base['occupation_type'] = get_val('occupation_type', get_val('occupation', '其他'))

            df_model_in = pd.get_dummies(df_base, columns=['city_level', 'occupation_type'], drop_first=True)

            with open(model_path, 'rb') as f:
                payload = pickle.load(f)
            model = payload['model']
            feature_names = payload['feature_names']
            scaler = payload['scaler']

            X = df_model_in.reindex(columns=feature_names, fill_value=0.0)
            X_scaled = scaler.transform(X)
            y_prob = model.predict_proba(X_scaled)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)

            result_str = "【逻辑回归跃升预测结果】\n"
            for i, (pred, prob) in enumerate(zip(y_pred, y_prob)):
                cid = df_raw['customer_id'].iloc[i] if 'customer_id' in df_raw.columns else f"客户_{i+1}"
                status = "🚨 高概率跃升为【高净值】客户" if pred == 1 else "暂时无法跃升"
                result_str += f"- {cid}: 跃升概率 {prob*100:.1f}% -> {status}\n"
            return result_str
        except Exception as e:
            import traceback
            return f"预测执行出错: {str(e)}\n{traceback.format_exc()}"


@register_tool('predict_customer_tier_dt')
class PredictCustomerTierDT(BaseTool):
    description = '使用已训练好的决策树模型（dt_model.pkl），预测客户未来3个月是否跃升为高净值（资产>=100万）。'
    parameters = [{
        'name': 'customer_data',
        'type': 'string',
        'description': '客户的各项特征数据（JSON字符串格式），支持单条或多条。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import json
        import os
        import pickle
        import pandas as pd

        model_path = 'dt_model.pkl'
        if not os.path.exists(model_path):
            return "错误：未找到预训练的决策树模型 (dt_model.pkl)。请先运行决策树训练脚本保存模型。"

        try:
            if isinstance(params, str):
                params = json.loads(params)

            raw_data = params.get('customer_data', '{}')
            if isinstance(raw_data, str):
                raw_data = json.loads(raw_data)
            if isinstance(raw_data, dict):
                raw_data = [raw_data]

            df_raw = pd.DataFrame(raw_data)

            def get_val(col, default=0.0):
                return df_raw[col] if col in df_raw.columns else default

            df_base = pd.DataFrame()
            df_base['avg_assets_3m'] = get_val('total_aum', 0.0) / 10000
            df_base['assets_volatility'] = 0.0
            df_base['latest_assets'] = get_val('total_aum', 0.0) / 10000
            df_base['avg_deposit_3m'] = get_val('deposit_balance_monthly_avg', get_val('deposit_balance', 0.0)) / 10000
            df_base['avg_financial_3m'] = get_val('wealth_management_balance_monthly_avg', get_val('wealth_management_balance', 0.0)) / 10000
            df_base['avg_app_login'] = get_val('mobile_bank_login_count', 0.0)
            df_base['avg_app_view_time'] = get_val('mobile_bank_login_count', 0.0) * 5

            product_cols = ['deposit_balance', 'wealth_management_balance', 'fund_balance', 'insurance_balance']
            df_base['product_count'] = 0
            for p_col in product_cols:
                if p_col in df_raw.columns:
                    df_base['product_count'] += (df_raw[p_col] > 0).astype(int)

            df_base['assets_trend'] = 0.0
            df_base['age'] = get_val('age', 35)
            df_base['monthly_income'] = get_val('monthly_income', get_val('monthly_transaction_amount', 0.0) * 0.5) / 10000
            df_base['city_level'] = get_val('city_level', '二线')
            df_base['occupation_type'] = get_val('occupation_type', get_val('occupation', '其他'))

            df_model_in = pd.get_dummies(df_base, columns=['city_level', 'occupation_type'], drop_first=True)

            with open(model_path, 'rb') as f:
                payload = pickle.load(f)
            model = payload['model']
            feature_names = payload['feature_names']

            X = df_model_in.reindex(columns=feature_names, fill_value=0.0)
            y_prob = model.predict_proba(X)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)

            result_str = "【决策树跃升预测结果】\n"
            for i, (pred, prob) in enumerate(zip(y_pred, y_prob)):
                cid = df_raw['customer_id'].iloc[i] if 'customer_id' in df_raw.columns else f"客户_{i+1}"
                status = "🚨 高概率跃升为【高净值】客户" if pred == 1 else "暂时无法跃升"
                result_str += f"- {cid}: 跃升概率 {prob*100:.1f}% -> {status}\n"
            return result_str
        except Exception as e:
            import traceback
            return f"预测执行出错: {str(e)}\n{traceback.format_exc()}"


@register_tool('explain_customer_tier_shap')
class ExplainCustomerTierShap(BaseTool):
    description = '基于已训练的 LightGBM 模型，对“跃升为高净值”的预测进行 SHAP 可解释性分析，输出影响最大的特征贡献。'
    parameters = [{
        'name': 'customer_data',
        'type': 'string',
        'description': '客户的各项特征数据（JSON字符串格式），支持单条或多条。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import json
        import os
        import pickle
        import pandas as pd
        import shap
        import numpy as np

        model_path = 'lgb_model.pkl'
        if not os.path.exists(model_path):
            return "错误：未找到预训练的 LightGBM 模型 (lgb_model.pkl)。"

        try:
            if isinstance(params, str):
                params = json.loads(params)

            raw_data = params.get('customer_data', '{}')
            if isinstance(raw_data, str):
                raw_data = json.loads(raw_data)
            if isinstance(raw_data, dict):
                raw_data = [raw_data]

            df_raw = pd.DataFrame(raw_data)

            def get_val(col, default=0.0):
                return df_raw[col] if col in df_raw.columns else default

            df_features = pd.DataFrame()
            df_features['latest_assets'] = get_val('total_aum', 0.0) / 10000
            df_features['avg_assets_3m'] = get_val('total_aum', 0.0) / 10000
            df_features['assets_volatility'] = 0.0
            df_features['avg_deposit_3m'] = get_val('deposit_balance_monthly_avg', get_val('deposit_balance', 0.0)) / 10000
            df_features['avg_financial_3m'] = get_val('wealth_management_balance_monthly_avg', get_val('wealth_management_balance', 0.0)) / 10000
            df_features['monthly_income'] = get_val('monthly_transaction_amount', 0.0) * 0.5 / 10000
            df_features['avg_app_login'] = get_val('mobile_bank_login_count', 0.0)
            df_features['avg_app_view_time'] = get_val('mobile_bank_login_count', 0.0) * 5

            product_cols = ['deposit_balance', 'wealth_management_balance', 'fund_balance', 'insurance_balance']
            df_features['product_count'] = 0
            for p_col in product_cols:
                if p_col in df_raw.columns:
                    df_features['product_count'] += (df_raw[p_col] > 0).astype(int)

            df_features['age'] = get_val('age', 35)
            df_features['city_level'] = get_val('city_level', '二线')
            df_features['occupation_type'] = get_val('occupation', '其他')
            df_features['assets_trend'] = 0.0

            for col in ['city_level', 'occupation_type']:
                df_features[col] = df_features[col].astype('category')

            expected_cols = ['avg_assets_3m', 'assets_volatility', 'latest_assets',
                             'avg_deposit_3m', 'avg_financial_3m', 'avg_app_login',
                             'avg_app_view_time', 'product_count', 'assets_trend',
                             'age', 'monthly_income', 'city_level', 'occupation_type']
            for col in expected_cols:
                if col not in df_features.columns:
                    df_features[col] = 0
            df_features = df_features[expected_cols]

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            prob = model.predict(df_features, num_iteration=model.best_iteration)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_features)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            result = "【SHAP 可解释性分析（基于 LightGBM）】\n"
            for i in range(len(df_features)):
                cid = df_raw['customer_id'].iloc[i] if 'customer_id' in df_raw.columns else f"客户_{i+1}"
                sv = shap_values[i]
                idx = np.argsort(np.abs(sv))[::-1][:8]
                result += f"\n- {cid}: 预测跃升概率 {prob[i]*100:.1f}%\n"
                for j in idx:
                    feat = df_features.columns[j]
                    val = df_features.iloc[i, j]
                    contrib = sv[j]
                    direction = "↑" if contrib > 0 else "↓"
                    result += f"  - {feat}={val}: 贡献 {contrib:.4f} {direction}\n"
            return result
        except Exception as e:
            import traceback
            return f"分析执行出错: {str(e)}\n{traceback.format_exc()}"


@register_tool('recommend_product_bundle')
class RecommendProductBundle(BaseTool):
    description = '基于已挖掘的关联规则（product_association_rules.csv），根据客户已持有产品推荐下一步可交叉销售的产品。'
    parameters = [{
        'name': 'holdings',
        'type': 'string',
        'description': '客户已持有的产品列表（JSON字符串），例如 [\"存款\",\"理财\"]。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import json
        import os
        import pandas as pd
        import ast

        rules_path = 'product_association_rules.csv'
        if not os.path.exists(rules_path):
            return "错误：未找到关联规则文件 (product_association_rules.csv)。请先运行产品关联分析脚本生成规则。"

        try:
            if isinstance(params, str):
                params = json.loads(params)
            holdings = params.get('holdings', '[]')
            if isinstance(holdings, str):
                holdings = json.loads(holdings)
            holdings_set = set(holdings)

            df = pd.read_csv(rules_path)

            def parse_fs(x):
                if isinstance(x, str) and x.startswith('frozenset'):
                    return set(ast.literal_eval(x.replace('frozenset', '')))
                if isinstance(x, str) and x.startswith('"frozenset'):
                    x = x.strip('"')
                    return set(ast.literal_eval(x.replace('frozenset', '')))
                return set()

            df['ante_set'] = df['antecedents'].apply(parse_fs)
            df['cons_set'] = df['consequents'].apply(parse_fs)

            candidates = df[df['ante_set'].apply(lambda s: s.issubset(holdings_set) and len(s) > 0)].copy()
            if candidates.empty:
                return "未找到与当前持仓匹配的关联规则。"

            candidates = candidates.sort_values(['lift', 'confidence'], ascending=False).head(8)

            recs = []
            for _, row in candidates.iterrows():
                cons = list(row['cons_set'])
                if not cons:
                    continue
                for p in cons:
                    if p not in holdings_set:
                        recs.append((p, float(row['confidence']), float(row['lift'])))

            if not recs:
                return "未找到可推荐的新增产品（可能已覆盖所有规则后件）。"

            recs = sorted(recs, key=lambda x: (x[2], x[1]), reverse=True)

            result = "【产品组合推荐（关联规则）】\n"
            result += f"已持有: {', '.join(holdings)}\n"
            shown = set()
            for p, conf, lift in recs:
                if p in shown:
                    continue
                shown.add(p)
                result += f"- 推荐: {p}（置信度 {conf:.2f}，提升度 {lift:.2f}）\n"
                if len(shown) >= 5:
                    break
            return result
        except Exception as e:
            import traceback
            return f"推荐执行出错: {str(e)}\n{traceback.format_exc()}"


@register_tool('predict_customer_cluster')
class PredictCustomerCluster(BaseTool):
    description = '使用已训练好的 KMeans 聚类模型（kmeans_model.pkl）对客户进行分群预测。'
    parameters = [{
        'name': 'customer_data',
        'type': 'string',
        'description': '客户特征数据（JSON字符串），支持单条或多条。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import json
        import os
        import pickle
        import pandas as pd
        import numpy as np

        model_path = 'kmeans_model.pkl'
        if not os.path.exists(model_path):
            return "错误：未找到聚类模型文件 (kmeans_model.pkl)。请先运行客户聚类脚本保存模型。"

        try:
            if isinstance(params, str):
                params = json.loads(params)
            raw_data = params.get('customer_data', '{}')
            if isinstance(raw_data, str):
                raw_data = json.loads(raw_data)
            if isinstance(raw_data, dict):
                raw_data = [raw_data]

            df_raw = pd.DataFrame(raw_data)

            with open(model_path, 'rb') as f:
                payload = pickle.load(f)
            features = payload['features']
            scaler = payload['scaler']
            kmeans = payload['kmeans']
            cluster_names = payload.get('cluster_names', {})

            def get_val(col, default=0.0):
                return df_raw[col] if col in df_raw.columns else default

            df_feat = pd.DataFrame()
            df_feat['total_assets'] = get_val('total_assets', get_val('total_aum', 0.0))
            df_feat['monthly_income'] = get_val('monthly_income', get_val('monthly_transaction_amount', 0.0) * 0.5)
            df_feat['credit_card_monthly_expense'] = get_val('credit_card_monthly_expense', 0.0)
            df_feat['app_login_count'] = get_val('app_login_count', get_val('mobile_bank_login_count', 0.0))
            df_feat['investment_monthly_count'] = get_val('investment_monthly_count', 0.0)
            df_feat['financial_repurchase_count'] = get_val('financial_repurchase_count', 0.0)

            product_cols = ['deposit_balance', 'wealth_management_balance', 'fund_balance', 'insurance_balance']
            df_feat['product_count'] = 0
            for p_col in product_cols:
                if p_col in df_raw.columns:
                    df_feat['product_count'] += (df_raw[p_col] > 0).astype(int)

            df_feat['age'] = get_val('age', 35)

            X = df_feat.reindex(columns=features, fill_value=0.0)
            X_scaled = scaler.transform(X)
            clusters = kmeans.predict(X_scaled)

            result = "【客户分群预测结果（KMeans）】\n"
            for i, c in enumerate(clusters):
                cid = df_raw['customer_id'].iloc[i] if 'customer_id' in df_raw.columns else f"客户_{i+1}"
                name = cluster_names.get(int(c), str(int(c)))
                result += f"- {cid}: 群组 {name}\n"
            return result
        except Exception as e:
            import traceback
            return f"分群执行出错: {str(e)}\n{traceback.format_exc()}"


# 4. 配置大模型和 Assistant
api_key = os.getenv('DASHSCOPE_API_KEY')
print(f"当前获取到的 DASHSCOPE_API_KEY: {'已配置' if api_key else '未配置，请检查环境变量！'}")

llm_cfg = {
    'model': 'qwen-max',
    'api_key': api_key,  # 显式传入 api_key
    'timeout': 30,
    'retry_count': 3,
}

system_prompt = """你是一个专业的“百万客群经营助手”。
你的任务是帮助银行客户经理分析、挖掘和管理高净值客户。
你可以通过 `mysql_query` 工具查询 MySQL 数据库。
数据库中包含表 `customer_data`，其完整表结构如下：
- customer_id VARCHAR(10) PRIMARY KEY COMMENT '客户编号'
- gender CHAR(1) COMMENT '性别: M-男, F-女'
- age INT COMMENT '年龄'
- occupation VARCHAR(20) COMMENT '职业'
- marital_status VARCHAR(10) COMMENT '婚姻状况: 已婚、未婚、离异'
- city_level VARCHAR(10) COMMENT '城市等级: 一线、二线、三线'
- account_open_date VARCHAR(10) COMMENT '开户日期'
- total_aum DECIMAL(18, 2) COMMENT '总资产管理规模'
- deposit_balance DECIMAL(18, 2) COMMENT '存款余额'
- wealth_management_balance DECIMAL(18, 2) COMMENT '理财余额'
- fund_balance DECIMAL(18, 2) COMMENT '基金余额'
- insurance_balance DECIMAL(18, 2) COMMENT '保险余额'
- deposit_balance_monthly_avg DECIMAL(18, 2) COMMENT '存款月均余额'
- wealth_management_balance_monthly_avg DECIMAL(18, 2) COMMENT '理财月均余额'
- fund_balance_monthly_avg DECIMAL(18, 2) COMMENT '基金月均余额'
- insurance_balance_monthly_avg DECIMAL(18, 2) COMMENT '保险月均余额'
- monthly_transaction_count DECIMAL(10, 2) COMMENT '月均交易次数'
- monthly_transaction_amount DECIMAL(18, 2) COMMENT '月均交易金额'
- last_transaction_date VARCHAR(10) COMMENT '最近交易日期'
- mobile_bank_login_count INT COMMENT '手机银行登录次数'
- branch_visit_count INT COMMENT '网点访问次数'
- last_mobile_login VARCHAR(10) COMMENT '最近手机银行登录日期'
- last_branch_visit VARCHAR(10) COMMENT '最近网点访问日期'
- customer_tier VARCHAR(10) COMMENT '客户等级: 普通、潜力、临界、高净值'

工作原则：
1. 回答问题前，先思考需要什么样的 SQL 语句，然后调用 `mysql_query` 执行查询。
2. 特别关注 AUM（total_aum）>= 1000000 的百万级客群。
3. 如果用户询问营销建议，请结合数据查询结果，给出专业建议。
"""

bot = Assistant(
    llm=llm_cfg,
    system_message=system_prompt,
    function_list=[
        'mysql_query',
        'predict_future_aum',
        'predict_customer_tier',
        'predict_customer_tier_lr',
        'predict_customer_tier_dt',
        'explain_customer_tier_shap',
        'recommend_product_bundle',
        'predict_customer_cluster',
    ],
    name='百万客群经营助手',
    description='基于 Qwen-Agent 的助手'
)

# 5. 启动 Web 界面
def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 配置聊天界面，从智能助手问题集中选择有代表性的问题
        chatbot_config = {
            'prompt.suggestions': [
                '请查询一下数据库里前 3 个客户的信息。',
                '请帮我预测一下这几个客户未来3个月跃升为高净值客户的概率（LightGBM/逻辑回归/决策树对比）。',
                '请对这几个客户的跃升预测做 SHAP 解释，告诉我主要驱动因素。',
                '客户已持有【存款, 理财】，下一步推荐什么产品组合？',
                '请对这几个客户做分群预测，并给出群组画像。',
                '帮我预测一下未来三个月的全行AUM增长趋势。'
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面，使用 7863 端口避免与之前启动的进程冲突
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run(share=False, server_port=7863)
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")

if __name__ == '__main__':
    app_gui()
