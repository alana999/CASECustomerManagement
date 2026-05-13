# 🏦 百万客群智能经营系统 (AI Customer Management System)

基于 **FastAPI + Streamlit + 多 Agent 协同 + RAG + 机器学习** 构建的银行客户智能经营与洞察系统。专为银行客户经理 (RM) 设计，旨在通过数据驱动精准营销，提升百万级高净值客群的转化率，降低流失率与营销成本。

## ✨ 核心功能

*   **📊 可视化大屏 (Dashboard)**：秒级加载全行资产分层、城市分布、客群画像等宏观指标。
*   **🤖 对话式 BI (智能助手)**：通过自然语言交互，完成复杂的数据查询与多维量化分析。
*   **🧠 机器学习深度预测**：
    *   **LightGBM/逻辑回归/决策树**：预测客户资产跃升概率，识别高潜力客户。
    *   **KMeans 聚类**：自动客户分群，支持精准营销。
    *   **ARIMA 时序分析**：预测全行未来 AUM（资产管理规模）趋势。
    *   **SHAP 归因分析**：为模型预测结果提供特征重要性解释，打破“黑盒”模型，增强业务说服力。
*   **📚 RAG 营销话术生成**：结合 ChromaDB 向量数据库，根据预测结果语义检索企业话术库，动态生成个性化、有温度的 1V1 沟通脚本（防大模型幻觉）。

## 🏗️ 架构设计

本项目采用 **MVC 前后端分离架构** 与 **多 Agent 协同工作流**，并针对极限资源环境（如 2GB 内存云主机）进行了深度的工程化与性能优化。

*   **View 层 (前端)**：Streamlit 多页面应用，提供流畅的 SSE (Server-Sent Events) 对话流式体验。
*   **Controller 层 (后端)**：FastAPI 提供 RESTful API 接口，解耦大屏渲染与智能对话请求。
*   **Agent 层 (多智能体编排)**：
    *   `Router Agent` (总控)：意图识别与任务路由。
    *   `Data Agent` (数据专家)：SQL 执行、调用 `.pkl` 离线模型进行量化分析推理。
    *   `Comm Agent` (沟通专家)：基于 RAG 检索知识库，重写并生成带有情绪价值的营销脚本。
*   **Model/Data 层 (数据与基座)**：
    *   **MySQL (MariaDB)**：存储客户基础与行为资产数据。
    *   **ChromaDB**：本地向量数据库，存储切块后的营销话术模板 (基于 `sentence-transformers` 嵌入)。
    *   **ML Models**：离线训练的 `.pkl` 模型，后端服务启动时**单例预加载**，实现毫秒级推理并防止 OOM。

## 🚀 极简上云部署指南 (以 Ubuntu 22.04 2GB RAM 为例)

本项目支持在极低配置的云服务器上平稳运行。

### 1. 环境准备
```bash
# 安装基础依赖与 MariaDB
sudo apt update
sudo apt install python3-venv python3-pip mariadb-server nginx unzip -y

# 配置 2GB Swap (防 OOM 核心操作)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 2. 数据库配置 (极低内存模式)
```bash
# 限制 MariaDB 内存占用
sudo sed -i '/\[mysqld\]/a innodb_buffer_pool_size = 128M\nmax_connections = 50\nperformance_schema = off' /etc/mysql/mariadb.conf.d/50-server.cnf
sudo systemctl restart mariadb

# 初始化数据库 (执行 SQL)
sudo mysql -e "CREATE DATABASE case_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
sudo mysql -e "CREATE USER 'case_user'@'localhost' IDENTIFIED BY 'Case_123456';"
sudo mysql -e "GRANT ALL PRIVILEGES ON case_db.* TO 'case_user'@'localhost'; FLUSH PRIVILEGES;"

# (可选) 导入本地数据
# sudo mysql case_db < bank_data.sql
```

### 3. 项目配置与依赖安装
```bash
# 1. 解压项目代码 (假设已上传 zip 包)
unzip case_project.zip -d /home/ubuntu/CASECustomerManagement
cd /home/ubuntu/CASECustomerManagement

# 2. 配置环境变量 (包含数据库密码与阿里云大模型 API Key)
sudo tee /etc/case-customer.env >/dev/null <<'EOF'
DB_HOST=127.0.0.1
DB_USER=case_user
DB_PASS=Case_123456
DB_NAME=case_db
DASHSCOPE_API_KEY=sk-your-api-key-here
EOF
sudo chmod 600 /etc/case-customer.env

# 3. 创建虚拟环境并安装核心依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 4. Systemd 守护进程与 Nginx 代理配置

为保证服务 24 小时稳定在线，采用 `systemd` 托管服务，并通过 `Nginx` 代理 80 端口。

```bash
# 1. 注册 FastAPI 后端服务
sudo tee /etc/systemd/system/case-backend.service >/dev/null <<'EOF'
[Unit]
Description=CASE Customer Management FastAPI Backend
After=network.target mariadb.service

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/CASECustomerManagement
EnvironmentFile=/etc/case-customer.env
ExecStart=/home/ubuntu/CASECustomerManagement/.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 2. 注册 Streamlit 前端服务
sudo tee /etc/systemd/system/case-frontend.service >/dev/null <<'EOF'
[Unit]
Description=CASE Customer Management Streamlit Frontend
After=network.target case-backend.service

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/CASECustomerManagement
EnvironmentFile=/etc/case-customer.env
ExecStart=/home/ubuntu/CASECustomerManagement/.venv/bin/streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 3. 启动后台服务
sudo systemctl daemon-reload
sudo systemctl enable --now case-backend
sudo systemctl enable --now case-frontend

# 4. 配置 Nginx 反向代理
sudo tee /etc/nginx/sites-available/case_customer >/dev/null <<'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
EOF

# 5. 重启 Nginx (确保云控制台安全组已放行 80 端口)
sudo ln -s /etc/nginx/sites-available/case_customer /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
```

此时，直接访问服务器公网 IP 即可使用系统！

## 📁 核心目录结构
```text
CASECustomerManagement/
├── backend/                  # FastAPI 后端目录
│   ├── agent/                # 多 Agent 协同模块 (Router, Tools, Prompts)
│   ├── core/                 # 核心配置与数据库连接单例
│   ├── models/               # 离线 .pkl 模型存储库 & 单例加载器
│   ├── routers/              # API 路由 (chat SSE, dashboard)
│   └── main.py               # 后端启动入口
├── frontend/                 # Streamlit 前端目录
│   ├── pages/                # 大屏与对话视图页面
│   └── app.py                # 前端主入口
├── chroma_db/                # (持久化) ChromaDB 向量知识库
├── requirements.txt          # 项目精简核心依赖
└── README.md
```

## 🛠️ 技术栈
*   **后端**: Python 3, FastAPI, Uvicorn, Pydantic
*   **前端**: Streamlit, Plotly, Pandas
*   **大模型 & 框架**: Qwen-Agent, 阿里云通义千问 API (qwen-max)
*   **数据与机器学习**: MySQL, Scikit-learn, LightGBM, Statsmodels (ARIMA), SHAP
*   **RAG 向量检索**: ChromaDB, Sentence-Transformers
*   **部署运维**: Nginx, Systemd, Linux Swap Tuning