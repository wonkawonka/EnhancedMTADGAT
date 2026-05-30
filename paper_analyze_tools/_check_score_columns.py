"""检查 test_output 的列结构和分数范围"""
import pandas as pd
from pathlib import Path

base = Path(r'D:\作业\论文复现\基线模型\mtad-gat-pytorch\kaggle离线output\ch3_main_results')
for name in ['msl_mtadgat_baseline', 'msl_c3_full', 'smap_mtadgat_baseline', 'smap_c3_full']:
    p = base / name / 'test_output.pkl'
    df = pd.read_pickle(p)
    label_cols = [c for c in df.columns if 'label' in c.lower() or 'anomaly' in c.lower() or 'y_true' in c.lower() or 'Label' in c]
    print(f'{name}: shape={df.shape}')
    print(f'  columns: {list(df.columns)}')
    print(f'  A_Score_Global range: [{df["A_Score_Global"].min():.4f}, {df["A_Score_Global"].max():.4f}]')
    if label_cols:
        for lc in label_cols:
            print(f'  {lc} unique values: {df[lc].unique()}')
    print()
