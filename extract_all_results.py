import json
import os

def extract_smap_msl(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if 'bf_result' in data and data['bf_result']:
                res = data['bf_result']
            elif 'pot_result' in data and data['pot_result']:
                res = data['pot_result']
            else:
                res = data.get('epsilon_result', {})
            
            return {
                'F1': res.get('f1', None),
                'Precision': res.get('precision', None),
                'Recall': res.get('recall', None),
                'TP': res.get('TP', None),
                'FP': res.get('FP', None),
                'FN': res.get('FN', None)
            }
    except Exception as e:
        return {}

results = []
base_dir = "kaggle离线output"
for trans_status in ["开了transformer", "没开transformer"]:
    status_dir = os.path.join(base_dir, trans_status)
    if not os.path.exists(status_dir):
        continue
        
    for exp_dir in os.listdir(status_dir):
        full_path = os.path.join(status_dir, exp_dir)
        if not os.path.isdir(full_path):
            continue
            
        parts = exp_dir.split('_')
        if len(parts) < 6:
            continue
            
        dataset = parts[1].upper()
        if dataset not in ["SMAP", "MSL"]:
            continue
            
        ms_mode = "none"
        if "base" in exp_dir or "none" in exp_dir:
            ms_mode = "none"
        elif "ms" in exp_dir or "basic" in exp_dir:
            ms_mode = "basic"
            
        feat_trans = "ON" if "feattrans-on" in exp_dir else "OFF"
        
        row = {
            "Dataset": dataset,
            "Multi-Scale": ms_mode,
            "FeatTrans": feat_trans
        }
        
        metrics_file = os.path.join(full_path, "summary_metrics.json")
        if os.path.exists(metrics_file):
            metrics = extract_smap_msl(metrics_file)
            row.update(metrics)
            results.append(row)

print(f"{'Dataset':<10} | {'FeatTrans':<10} | {'Multi-Scale':<12} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'FP (误报)':<10}")
print("-" * 85)
results.sort(key=lambda x: (x['Dataset'], x['FeatTrans'], x['Multi-Scale']))

for r in results:
    f1 = f"{r.get('F1', 0):.4f}" if r.get('F1') is not None else "N/A"
    prec = f"{r.get('Precision', 0):.4f}" if r.get('Precision') is not None else "N/A"
    rec = f"{r.get('Recall', 0):.4f}" if r.get('Recall') is not None else "N/A"
    fp = str(int(r.get('FP', 0))) if r.get('FP') is not None else "N/A"
    
    print(f"{r['Dataset']:<10} | {r['FeatTrans']:<10} | {r['Multi-Scale']:<12} | {f1:<10} | {prec:<10} | {rec:<10} | {fp:<10}")
