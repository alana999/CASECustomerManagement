import os
from qwen_agent.agents import Assistant, Router
import backend.agent.tools

def get_data_agent_prompt() -> str:
    return """你是一个专业的“数据与分析专家 Agent”。
    你的核心任务是查询 MySQL 数据库，并使用多种预测模型进行量化分析。
    你只负责给出客观的数据和分析结论，不需要生成对客户的沟通话术。

    数据库包含表 `customer_data`，完整表结构如下：
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
    - customer_tier VARCHAR(10) COMMENT '客户等级'

    工作原则：
    1. 回答问题前，先思考需要什么样的 SQL 语句，然后调用 `mysql_query` 执行查询。
    2. 特别关注 AUM（total_aum）>= 1000000 的百万级客群。
    3. 如果需要预测，调用对应的预测工具（如 predict_customer_tier_lr 等）给出结论。
    """

def get_comm_agent_prompt() -> str:
    return """你是一个专业的“营销沟通专家 Agent”。
    你的核心任务是根据前端传来的客户数据或分析结论，撰写高转化率的营销话术和沟通脚本。
    你不需要直接查询数据库，你的职责是将冷冰冰的数据转化为有温度的沟通方案。

    工作原则：
    1. 当你需要生成营销脚本、挽留建议或安抚话术时，必须先调用 `retrieve_marketing_script` 工具从企业知识库中检索标准话术模板。
    2. 严格参考检索到的模板风格和切入点，结合具体的客户画像（如性别、资产亏损情况等），生成个性化的 1V1 沟通脚本。
    3. 语气要体现银行客户经理的专业、同理心与服务意识。
    """

def create_multi_agent() -> Router:
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        raise ValueError("未配置 DASHSCOPE_API_KEY 环境变量！")

    llm_cfg = {
        'model': 'qwen-max',
        'api_key': api_key,
        'timeout': 30,
        'retry_count': 3,
    }

    # 1. 创建数据分析专家 Agent
    data_agent = Assistant(
        llm=llm_cfg,
        system_message=get_data_agent_prompt(),
        function_list=[
            'mysql_query', 
            'predict_future_aum', 
            'predict_customer_tier',
            'predict_customer_tier_lr',
            'predict_customer_tier_dt',
            'explain_customer_tier_shap',
            'recommend_product_bundle',
            'predict_customer_cluster'
        ],
        name='数据分析专家',
        description='擅长通过执行 SQL 查询数据库，调用各类机器学习模型（分类、聚类、时序、关联分析）进行深度量化分析，输出客观的数据洞察和预测结论。'
    )

    # 2. 创建营销沟通专家 Agent
    comm_agent = Assistant(
        llm=llm_cfg,
        system_message=get_comm_agent_prompt(),
        function_list=['retrieve_marketing_script'],
        name='营销沟通专家',
        description='擅长根据客户经理的意图和已有的数据分析结论，从企业 RAG 知识库中检索标准话术模板，并撰写个性化、有温度的 1V1 营销沟通脚本或挽留方案。'
    )

    # 3. 创建路由总控 Agent (Router)
    # Router 会根据用户的输入，自动将任务分发给最合适的下级专家 Agent
    router_agent = Router(
        llm=llm_cfg,
        agents=[data_agent, comm_agent],
        name='百万客群总管',
        description='我是百万客群经营的总控管家，负责理解您的意图，并协调后台的数据分析专家和营销沟通专家为您提供全方位的智能服务。'
    )

    return router_agent

# 导出全局的路由 Agent 作为入口
bot = create_multi_agent()
