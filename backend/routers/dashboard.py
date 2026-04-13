from fastapi import APIRouter, HTTPException
from backend.core.db_client import db_client
import pandas as pd

router = APIRouter()

@router.get("/stats/kpi")
async def get_kpi_stats():
    """获取核心 KPI 数据（总资产、总客户数、高净值客户数）"""
    try:
        # 查询总客户数和总 AUM
        sql_total = "SELECT COUNT(*) as total_customers, SUM(total_aum) as total_aum FROM customer_data"
        total_data = db_client.execute_query(sql_total)[0]
        
        # 暂时用 AUM >= 1000000 来统计高净值（如果 customer_tier 为空）
        sql_hwni = "SELECT COUNT(*) as hwni_count FROM customer_data WHERE total_aum >= 1000000"
        hwni_data = db_client.execute_query(sql_hwni)[0]

        return {
            "total_customers": total_data.get("total_customers", 0),
            "total_aum": float(total_data.get("total_aum") or 0.0),
            "hwni_count": hwni_data.get("hwni_count", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/asset_distribution")
async def get_asset_distribution():
    """获取资产结构分布（存款、理财、基金、保险）"""
    try:
        sql = """
        SELECT 
            SUM(deposit_balance) as deposit,
            SUM(wealth_management_balance) as wealth,
            SUM(fund_balance) as fund,
            SUM(insurance_balance) as insurance
        FROM customer_data
        """
        data = db_client.execute_query(sql)[0]
        
        # 组装成前端需要的列表格式
        result = [
            {"name": "存款", "value": float(data.get("deposit") or 0)},
            {"name": "理财", "value": float(data.get("wealth") or 0)},
            {"name": "基金", "value": float(data.get("fund") or 0)},
            {"name": "保险", "value": float(data.get("insurance") or 0)}
        ]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/city_distribution")
async def get_city_distribution():
    """获取城市等级分布"""
    try:
        sql = "SELECT city_level, COUNT(*) as count FROM customer_data GROUP BY city_level"
        data = db_client.execute_query(sql)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
