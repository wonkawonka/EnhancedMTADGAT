"""Highlight displayed best values in the neural development reports."""
NOTE = '粗体表示各表内同一数据集、同一指标的显示值最优；AP、AUROC、F1、Recall越高越好，FPR越低越好。显示精度下并列最优均加粗，不表示统计显著性。'


def bold_best_metrics(markdown):
    lines = markdown.splitlines()
    changed = False
    index = 0
    while index < len(lines):
        if not lines[index].startswith('|'):
            index += 1
            continue
        end = index
        while end < len(lines) and lines[end].startswith('|'):
            end += 1
        headers = [s.strip() for s in lines[index].strip('|').split('|')]
        rows = [[s.strip() for s in line.strip('|').split('|')] for line in lines[index+2:end]]
        triplets = len(headers) > 1 and all('AP / AUROC / F1' in h for h in headers[1:])
        alarms = headers[0] == '数据集' and any('Recall' in h for h in headers) and '测试FPR' in headers
        if rows and (triplets or alarms):
            groups = {}
            for i, row in enumerate(rows):
                groups.setdefault(row[0] if alarms else 'all', []).append(i)
            columns = range(1, len(headers)) if triplets else [j for j,h in enumerate(headers) if 'Recall' in h or h == '测试FPR']
            for members in groups.values():
                for col in columns:
                    values = {i: [float(v.strip().replace('**', '')) for v in rows[i][col].split(' / ')] for i in members}
                    best = [(min if 'FPR' in headers[col] else max)(values[i][j] for i in members) for j in range(len(values[members[0]]))]
                    for i in members:
                        tokens = rows[i][col].replace('**', '').split(' / ')
                        rows[i][col] = ' / '.join('**'+token+'**' if values[i][j] == best[j] else token for j,token in enumerate(tokens))
            lines[index+2:end] = ['| ' + ' | '.join(row) + ' |' for row in rows]
            changed = True
        index = end
    result = '\n'.join(lines) + ('\n' if markdown.endswith('\n') else '')
    if changed and NOTE not in result:
        first_table = result.index('\n|')
        result = result[:first_table] + '\n' + NOTE + '\n' + result[first_table:]
    return result
