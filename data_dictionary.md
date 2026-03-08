# 数据字典

本文档记录了 `customer_base.csv` 和 `customer_behavior_assets.csv` 数据表的字段含义。

## 1. customer_base.csv (客户基础信息表)

该表包含了客户的基本人口统计学特征及银行相关的基础信息。

| 字段名 | 中文含义 | 备注 |
| :--- | :--- | :--- |
| `customer_id` | 客户ID | 唯一标识符，用于关联其他表 |
| `name` | 姓名 | 客户姓名 |
| `age` | 年龄 | 客户年龄 |
| `gender` | 性别 | 男/女 |
| `occupation` | 职业 | 客户的具体职业名称 (如：律师、教师) |
| `occupation_type` | 职业类型 | 职业的分类 (如：专业人士、事业单位) |
| `monthly_income` | 月收入 | 客户每月的收入金额 |
| `open_account_date` | 开户日期 | 客户首次开户的日期 |
| `lifecycle_stage` | 生命周期阶段 | 客户所处的生命周期阶段 (如：新客户、成熟客户、忠诚客户) |
| `marriage_status` | 婚姻状况 | 已婚/未婚 |
| `city_level` | 城市等级 | 客户所在城市的等级 (如：一线城市) |
| `branch_name` | 分行名称 | 客户所属的银行分行名称 |

## 2. customer_behavior_assets.csv (客户行为与资产信息表)

该表记录了客户的资产持有情况、交易行为以及与银行的互动数据。

| 字段名 | 中文含义 | 备注 |
| :--- | :--- | :--- |
| `id` | 记录ID | 本表记录的唯一标识符 |
| `customer_id` | 客户ID | 关联 `customer_base.csv` 的外键 |
| `total_assets` | 总资产 | 客户在银行的总资产金额 |
| `deposit_balance` | 存款余额 | 存款账户的余额 |
| `financial_balance` | 理财余额 | 理财产品的持有金额 |
| `fund_balance` | 基金余额 | 基金产品的持有金额 |
| `insurance_balance` | 保险余额 | 保险产品的持有金额 |
| `asset_level` | 资产等级 | 根据总资产划分的等级 (如：50万以下) |
| `deposit_flag` | 是否持有存款 | 1: 是, 0: 否 |
| `financial_flag` | 是否持有理财 | 1: 是, 0: 否 |
| `fund_flag` | 是否持有基金 | 1: 是, 0: 否 |
| `insurance_flag` | 是否持有保险 | 1: 是, 0: 否 |
| `product_count` | 持有产品数量 | 客户持有的银行产品总数 |
| `financial_repurchase_count` | 理财复购次数 | 客户购买理财产品的复购次数 |
| `credit_card_monthly_expense` | 信用卡月消费额 | 信用卡每月的消费金额 |
| `investment_monthly_count` | 投资月交易次数 | 客户每月的投资交易次数 |
| `app_login_count` | APP登录次数 | 客户登录银行APP的次数 |
| `app_financial_view_time` | APP理财浏览时长 | 客户在APP上浏览理财相关页面的时长 |
| `app_product_compare_count` | APP产品对比次数 | 客户在APP上进行产品对比的次数 |
| `last_app_login_time` | 最近一次APP登录时间 | 客户最后一次登录APP的具体时间 |
| `last_contact_time` | 最近一次联系时间 | 银行最近一次联系客户的时间 |
| `contact_result` | 联系结果 | 最近一次联系的结果 (如：成功、未接通) |
| `marketing_cool_period` | 营销冷静期 | 客户处于营销冷静期的结束时间或状态 |
| `stat_month` | 统计月份 | 数据统计的月份 |
