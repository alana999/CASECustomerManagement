import json
from typing import Union
import pandas as pd
from qwen_agent.tools.base import BaseTool, register_tool

# 引入我们刚才重构的单例
from backend.core.db_client import db_client
from backend.models.ml_manager import ml_manager

# ---------------------------------------------------------
# 1. 数据库查询工具
# ---------------------------------------------------------
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
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {'sql_query': params}
        
        sql = params.get('sql_query', '')
        try:
            # 使用重构后的 db_client
            result = db_client.execute_query(sql)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"SQL执行出错: {str(e)}"

# ---------------------------------------------------------
# 2. ARIMA AUM预测工具
# ---------------------------------------------------------
@register_tool('predict_future_aum')
class PredictFutureAUM(BaseTool):
    description = '使用已训练好的 ARIMA 模型，预测全行未来一个季度（3个月）的 AUM（总资产管理规模）增长趋势。不需要传入任何参数。'
    parameters = []

    def call(self, params: Union[str, dict], **kwargs) -> str:
        # 从统一的 ml_manager 获取模型，不再需要每次 open() 和 pickle.load()
        model = ml_manager.get_model('arima')
        if not model:
            return "错误：未找到预训练的 ARIMA 模型。"
            
        try:
            forecast = model.get_forecast(steps=3)
            forecast_mean = forecast.predicted_mean
            
            result_str = "【ARIMA 模型预测结果】\n"
            for date, value in forecast_mean.items():
                result_str += f"- {date.strftime('%Y-%m')}: 预计达到 {value:.2f} 亿元\n"
            
            current_aum = model.data.endog[-1]
            future_aum = forecast_mean.iloc[-1]
            growth_rate = (future_aum - current_aum) / current_aum * 100
            
            result_str += f"\n当前最新 AUM: {current_aum:.2f} 亿元\n"
            result_str += f"预计季度末 AUM: {future_aum:.2f} 亿元\n"
            result_str += f"预计季度环比增长率: {growth_rate:.2f}%\n"
            
            return result_str
        except Exception as e:
            return f"预测执行出错: {str(e)}"

# TODO: 后续可以将 predict_customer_tier (LightGBM) 等工具按此模式继续补充进来...

# ---------------------------------------------------------
# 3. 知识库话术检索工具 (增强版 RAG，使用 ChromaDB)
# ---------------------------------------------------------
@register_tool('retrieve_marketing_script')
class RetrieveMarketingScript(BaseTool):
    description = '当需要向客户推销产品、安抚亏损或挽留流失客户时，调用此工具从企业向量知识库中语义检索最匹配的标准营销话术模板。'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '检索查询语句，用自然语言描述你需要找什么样的话术，例如："基金亏损安抚"、"向风险厌恶型客户推荐稳健理财"。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import os
        import json
        import chromadb
        from chromadb.utils import embedding_functions
        
        # 解析参数
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {'query': params}
        
        query = params.get('query', '')
        print(f"[{self.__class__.__name__}] 正在向向量数据库检索: {query}")
        
        # 定位向量数据库路径
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_dir = os.path.join(base_dir, 'chroma_db')
        
        if not os.path.exists(db_dir):
            return "错误：向量数据库不存在。请先运行 build_vector_db.py 构建知识库。"
            
        try:
            # 连接持久化的 ChromaDB
            client = chromadb.PersistentClient(path=db_dir)
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            
            collection = client.get_collection(
                name="marketing_scripts",
                embedding_function=sentence_transformer_ef
            )
            
            # 执行语义检索，取 Top 3 最相关的文本块
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if not results['documents'] or not results['documents'][0]:
                return f"未能在知识库中检索到与 '{query}' 相关的话术。请大模型自行基于银行业经验生成。"
                
            # 组装返回给 LLM 的字符串
            response_str = f"【向量知识库检索结果】(针对查询: {query})\n"
            for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
                # dist 越小越相似 (通常是欧氏距离或余弦距离)
                response_str += f"\n--- 来源文档 {i+1}: {meta.get('source', '未知')} (距离: {dist:.4f}) ---\n"
                response_str += doc + "\n"
                
            response_str += "\n💡 请严格参考以上【来源文档】中的话术风格和切入点，结合客户的具体数据，为其生成个性化的沟通脚本。"
            return response_str
            
        except Exception as e:
            return f"检索执行出错: {str(e)}"


@register_tool('predict_customer_tier')
class PredictCustomerTier(BaseTool):
    description = '使用已训练好的 LightGBM 模型，根据给定的客户资产和行为数据，预测该客户是否能在未来3个月内跃升为“高净值”客户（资产>=100万）。'
    parameters = [{
        'name': 'customer_data',
        'type': 'string',
        'description': '客户的各项特征数据（JSON字符串格式），支持单条或多条记录。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import json
        import numpy as np

        model = ml_manager.get_model('lightgbm')
        if not model:
            return "错误：未找到预训练的 LightGBM 模型 (lgb_model.pkl)。请先运行分类训练脚本保存模型。"

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

            expected_cols = [
                'avg_assets_3m',
                'assets_volatility',
                'latest_assets',
                'avg_deposit_3m',
                'avg_financial_3m',
                'avg_app_login',
                'avg_app_view_time',
                'product_count',
                'assets_trend',
                'age',
                'monthly_income',
                'city_level',
                'occupation_type',
            ]
            for col in expected_cols:
                if col not in df_features.columns:
                    df_features[col] = 0.0
            df_features = df_features[expected_cols]

            y_prob = model.predict(df_features, num_iteration=getattr(model, 'best_iteration', None))
            y_prob = np.array(y_prob).reshape(-1)
            y_pred = (y_prob > 0.5).astype(int)

            result_str = "【客户价值跃升预测结果（LightGBM）】\n"
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
        'description': '客户的各项特征数据（JSON字符串格式），支持单条或多条记录。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import json
        import numpy as np

        payload = ml_manager.get_model('logistic')
        if not payload or not isinstance(payload, dict):
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

            model = payload['model']
            feature_names = payload['feature_names']
            scaler = payload['scaler']

            X = df_model_in.reindex(columns=feature_names, fill_value=0.0)
            X_scaled = scaler.transform(X)
            y_prob = model.predict_proba(X_scaled)[:, 1]
            y_prob = np.array(y_prob).reshape(-1)
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
        'description': '客户的各项特征数据（JSON字符串格式），支持单条或多条记录。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import json
        import numpy as np

        payload = ml_manager.get_model('decision_tree')
        if not payload or not isinstance(payload, dict):
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

            model = payload['model']
            feature_names = payload['feature_names']

            X = df_model_in.reindex(columns=feature_names, fill_value=0.0)
            y_prob = model.predict_proba(X)[:, 1]
            y_prob = np.array(y_prob).reshape(-1)
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
        'description': '客户的各项特征数据（JSON字符串格式），支持单条或多条记录。',
        'required': True
    }]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        import json
        import numpy as np

        model = ml_manager.get_model('lightgbm')
        if not model:
            return "错误：未找到预训练的 LightGBM 模型 (lgb_model.pkl)。"

        try:
            try:
                import shap
            except Exception:
                return "错误：未安装 shap。请先安装 shap 后再使用可解释性分析。"

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

            expected_cols = [
                'avg_assets_3m',
                'assets_volatility',
                'latest_assets',
                'avg_deposit_3m',
                'avg_financial_3m',
                'avg_app_login',
                'avg_app_view_time',
                'product_count',
                'assets_trend',
                'age',
                'monthly_income',
                'city_level',
                'occupation_type',
            ]
            for col in expected_cols:
                if col not in df_features.columns:
                    df_features[col] = 0.0
            df_features = df_features[expected_cols]

            prob = model.predict(df_features, num_iteration=getattr(model, 'best_iteration', None))
            prob = np.array(prob).reshape(-1)

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
        import os
        import json
        import ast

        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {'holdings': params}

        holdings = params.get('holdings', '[]')
        if isinstance(holdings, str):
            holdings = json.loads(holdings)
        holdings_set = set(holdings)

        base_dir = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
        rules_path = os.path.join(base_dir, 'product_association_rules.csv')
        if not os.path.exists(rules_path):
            return "错误：未找到关联规则文件 (product_association_rules.csv)。请先运行产品关联分析脚本生成规则。"

        try:
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
        import numpy as np

        payload = ml_manager.get_model('kmeans')
        if not payload or not isinstance(payload, dict):
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
