import os
import pickle
import logging

logger = logging.getLogger(__name__)

class MLModelManager:
    """机器学习模型统一管理器（单例模式）
    在 FastAPI 启动时加载所有 .pkl，避免每次请求重复读取磁盘
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModelManager, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance._load_all_models()
        return cls._instance

    def _load_all_models(self):
        # 假设根目录在 backend 的上一层
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        model_files = {
            'lightgbm': 'lgb_model.pkl',
            'arima': 'arima_model.pkl',
            'logistic': 'lr_model.pkl',
            'decision_tree': 'dt_model.pkl',
            'kmeans': 'kmeans_model.pkl'
        }
        
        for name, filename in model_files.items():
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    logger.info(f"✅ 模型 {name} 加载成功.")
                except Exception as e:
                    logger.error(f"❌ 模型 {name} 加载失败: {e}")
            else:
                logger.warning(f"⚠️ 找不到模型文件: {path}")

    def get_model(self, model_name: str):
        """获取指定的模型实例"""
        return self.models.get(model_name)

# 暴露单例实例
ml_manager = MLModelManager()
