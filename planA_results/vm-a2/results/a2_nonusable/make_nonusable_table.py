"""Non-usable task accounting (HANDOFF_planBC.md 1.4). Run from planA_results/vm-a2/results:
    python3 a2_nonusable/make_nonusable_table.py
Reads a2_v3/a2_tasks.csv (125 rows) and a2_v3/a2_run_status.csv (129 rows; the only source for
the 4 requests no_tests tasks). Writes a2_nonusable/a2_nonusable.csv."""
import csv, collections
A = 'a2_v3'
t = {r['instance_id']: r for r in csv.DictReader(open(f'{A}/a2_tasks.csv'))}
s = {r['instance_id']: r for r in csv.DictReader(open(f'{A}/a2_run_status.csv'))}
order = {'clean': 0, 'collection_error': 1, 'import_error': 2, 'no_tests': 3}
rows = []
for i, r in s.items():
    if r['usable'] == 'True':
        continue
    tr = t.get(i)
    src = tr or r
    rc, prc = src['run_class'], src['post_run_class']
    ne, nc = src['n_f2p_expected'], src['n_f2p_confirmed']
    reason = {'collection_error': 'pre-fix pytest collection error',
              'import_error': 'pre-fix import error',
              'no_tests': 'no collectable tests'}.get(rc, rc)
    if rc == 'clean':
        reason = ('clean run, no fail-to-pass test expected' if ne == '0'
                  else 'clean run, expected fail-to-pass not confirmed')
    rows.append({'instance_id': i, 'repo': r['repo'], 'run_class': rc, 'post_run_class': prc,
                 'n_f2p_expected': ne, 'n_f2p_confirmed': nc, 'reason': reason,
                 'source': 'a2_tasks.csv' if tr else 'a2_run_status.csv'})
rows.sort(key=lambda x: (order.get(x['run_class'], 9), x['instance_id']))
with open('a2_nonusable/a2_nonusable.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
usable = [r for r in s.values() if r['usable'] == 'True']
print('non-usable', len(rows), dict(collections.Counter(r['run_class'] for r in rows)))
print('by reason', dict(collections.Counter(r['reason'] for r in rows)))
print('usable', len(usable), dict(collections.Counter(r['repo'] for r in usable)))
