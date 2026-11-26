# -*- coding: utf-8 -*-
import json, pathlib, sys
try:
    import json, pathlib, pandas as pd, numpy as np
    ROOT=pathlib.Path(__file__).resolve().parents[1]
    m={}
    o=pd.read_csv(ROOT/"data/real/online_retail_orders_sample.csv",parse_dates=["InvoiceDate"])
    o=o[o.Quantity>0]
    snap=o.InvoiceDate.max()
    rfm=o.groupby("CustomerID").agg(recency=("InvoiceDate",lambda s:(snap-s.max()).days),
        frequency=("InvoiceNo","nunique"),monetary=("Revenue","sum"))
    m["customers"]=int(len(rfm)); m["churn_rate_recency90"]=round(float((rfm.recency>90).mean()),4)
    m["avg_monetary"]=round(float(rfm.monetary.mean()),2)
    top=rfm.monetary.quantile(0.8); m["high_value_revenue_share"]=round(float(rfm.loc[rfm.monetary>=top,"monetary"].sum()/rfm.monetary.sum()),4)
    t=pd.read_csv(ROOT/"data/real/telco_churn.csv")
    t["TotalCharges"]=pd.to_numeric(t["TotalCharges"],errors="coerce")
    m["telco_churn_rate"]=round(float((t.Churn=="Yes").mean()),4)
    m["telco_churn_by_contract"]= {k:round(float((g.Churn=="Yes").mean()),4) for k,g in t.groupby("Contract")}
    print(json.dumps(m,ensure_ascii=False,indent=2))
    _root=pathlib.Path(__file__).resolve().parents[1]
    (_root/'outputs').mkdir(exist_ok=True)
    (_root/'reports').mkdir(exist_ok=True)
    (_root/'outputs/real_metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    NOTES='''# 真实公开数据集分析概览\n\n本仓库附带真实公开数据集（见 data/real/SOURCES.md），运行 `python src/real_data_analysis.py` 复现以下真实数据统计：\n\n```json\n'+json.dumps(m,ensure_ascii=False,indent=2)+'\n```\n\n> data/ 与 outputs/metrics.json 的主结果为脱敏/仿真案例，用于稳定演示方法；本文件使用公开真实数据集验证方法可运行。'''
    (_root/'reports/real-data-notes.md').write_text(NOTES,encoding='utf-8')
    print(json.dumps(m,ensure_ascii=False,indent=2))
except Exception as e:
    print('REAL_ANALYSIS_ERROR', type(e).__name__, e); sys.exit(1)
