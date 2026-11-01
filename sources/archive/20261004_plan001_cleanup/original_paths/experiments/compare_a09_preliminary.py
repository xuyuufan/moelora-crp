"""Reproducible two-seed development comparison, without final-test scoring."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from scrp.attention_data import file_sha256
from scrp.attention_evaluation import evaluate_episodes, load_attention_checkpoint, load_legacy_v1_checkpoint
from scrp.online_crp import load_episode_manifest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / 'archive/20261004_mainline_cleanup/legacy_moe_release'
OUT = ROOT / 'output/attention_multiscale_v1/preliminary_two_seed_20261004'
ATT_SEEDS = (20261011, 20261021)
MOE_SEEDS = (20260913, 20260923, 20260933)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save_json(path, record):
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def curves(report):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .18, 'legend.frameon': False,
        'savefig.dpi': 300, 'pdf.fonttype': 42})
    att_color, moe_color, base_color = '#0072B2', '#D55E00', '#555555'
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    keys = ('macro_mean_r_over_n', 'small_macro_mean_r_over_n', 'large_macro_mean_r_over_n')
    titles = ('All 25 seen domains', '21 structural domains', '4 large domains (N=100-150)')
    training = []
    for seed in ATT_SEEDS:
        path = ROOT / f'output/attention_multiscale_v1/runs/a09_joint_seed{seed}_n20_20261002/validation.jsonl'
        training.append([r for r in read_jsonl(path) if r['split'] == 'validation_seen'])
    steps = sorted(set(r['global_step'] for r in training[0]) & set(r['global_step'] for r in training[1]))
    maps = [{r['global_step']: r for r in rows} for rows in training]
    training_csv = []
    for ax, key, title in zip(axes, keys, titles):
        values = np.array([[m[s][key] for s in steps] for m in maps])
        for seed, vals in zip(ATT_SEEDS, values):
            ax.plot(steps, vals, color=att_color, alpha=.3, lw=.8)
        ax.plot(steps, values.mean(axis=0), color=att_color, lw=2, label='ATT-Joint (2-seed mean)')
        ax.fill_between(steps, values.min(axis=0), values.max(axis=0), color=att_color, alpha=.12)
        moe = [report['evaluations'][f'MoE_{s}']['validation_seen'][key] for s in MOE_SEEDS]
        base = report['evaluations']['H20']['validation_seen'][key]
        ax.axhline(np.mean(moe), color=moe_color, lw=1.8, ls='--', label='MoE-LoRA (3-seed mean)')
        ax.axhspan(min(moe), max(moe), color=moe_color, alpha=.12)
        ax.axhline(base, color=base_color, ls=':', lw=2, label='Original frozen H20')
        ax.set(title=title, xlabel='ATT-Joint training slots', ylabel='Domain macro mean R/N (lower is better)')
        for step, vals in zip(steps, values.T):
            training_csv.append({'group': title, 'slot': step, 'att_seed20261011': vals[0],
                'att_seed20261021': vals[1], 'att_mean': float(vals.mean()),
                'moe_mean': float(np.mean(moe)), 'h20': base})
    axes[0].legend(fontsize=8)
    fig.suptitle('H=20 | matched validation manifests | CPU FP32 greedy', fontsize=13)
    fig.text(.5, .012, 'Bands: seed min-max, not confidence intervals. Frozen baselines are evaluated once; their lines are reference levels.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94))
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig_training_comparison.{ext}')
    plt.close(fig)
    write_csv(OUT / 'training_curve.csv', training_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios':[2.8,1]})
    domains = report['evaluations']['H20']['validation_seen']['per_domain_macro_mean_r_over_n']
    structural = sorted((d for d in domains if d.startswith('table2')), key=lambda d: (int(d.split('_')[1][1:])*int(d.split('_')[2][1:]), d))
    large = sorted(d for d in domains if d.startswith('large'))
    for ax, ds, title in zip(axes, (structural, large), ('Structural domains, ordered by container count', 'Large seen geometries')):
        x = np.arange(len(ds))
        att = np.array([[report['evaluations'][f'ATT_{s}']['validation_seen']['per_domain_macro_mean_r_over_n'][d] for d in ds] for s in ATT_SEEDS])
        moe = np.array([[report['evaluations'][f'MoE_{s}']['validation_seen']['per_domain_macro_mean_r_over_n'][d] for d in ds] for s in MOE_SEEDS])
        base = [domains[d] for d in ds]
        ax.plot(x, att.mean(axis=0), '-o', color=att_color, ms=4, label='ATT-Joint (2 seeds)')
        ax.fill_between(x, att.min(axis=0), att.max(axis=0), color=att_color, alpha=.13)
        ax.plot(x, moe.mean(axis=0), '--s', color=moe_color, ms=4, label='MoE-LoRA (3 seeds)')
        ax.fill_between(x, moe.min(axis=0), moe.max(axis=0), color=moe_color, alpha=.13)
        ax.plot(x, base, ':^', color=base_color, ms=4, label='Original H20')
        labels = [d.replace('table2_', '').replace('large_', '').replace('_', '/') for d in ds]
        ax.set_xticks(x, labels, rotation=65, ha='right', fontsize=8)
        ax.set(title=title, xlabel=('Geometry: initial height h / stacks s' if ds == structural else 'Geometry: stacks s / capacity t'), ylabel='Mean R/N (lower is better)')
    axes[0].legend(fontsize=9)
    fig.suptitle('Selected best checkpoints | H=20 | same 620 validation episodes', fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'fig_scale_comparison.{ext}')
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(report, manifests):
    records = {}
    table = []
    for name in report['evaluations']:
        records[name] = {}
        for split in manifests:
            rows = read_jsonl(OUT / f'{name}_{split}.jsonl')
            assert len(rows) == len(manifests[split])
            expected = {e.canonical_id for e in manifests[split]}
            assert {r['canonical_id'] for r in rows} == expected
            assert len({r['canonical_id'] for r in rows}) == len(rows)
            assert all(r['completed'] and r['illegal_actions'] == 0 for r in rows)
            records[name][split] = {r['canonical_id']: r for r in rows}
            s = report['evaluations'][name][split]
            table.append({'method': name, 'split': split, **{k: s[k] for k in ('count','macro_mean_r_over_n','small_macro_mean_r_over_n','large_macro_mean_r_over_n','mean_r_over_n','completed_rate','illegal_actions')}})
    write_csv(OUT / 'comparison.csv', table)
    domain_table = []
    for split, episodes in manifests.items():
        domain_episodes = defaultdict(list)
        for e in episodes:
            domain_episodes[e.metadata['domain_id']].append(e)
        for domain, eps in sorted(domain_episodes.items()):
            domain_table.append({'split': split, 'domain': domain, 'count': len(eps),
                'N_min': min(e.num_containers for e in eps), 'N_max': max(e.num_containers for e in eps),
                **{name: report['evaluations'][name][split]['per_domain_macro_mean_r_over_n'][domain] for name in records}})
    write_csv(OUT / 'domain_comparison.csv', domain_table)
    comparisons = []
    rng = np.random.default_rng(20261101)
    for split, episodes in manifests.items():
        ids = [e.canonical_id for e in episodes]
        groups = defaultdict(list)
        for i, e in enumerate(episodes):
            groups[e.metadata['domain_id']].append(i)
        att = np.array([[records[f'ATT_{s}'][split][i]['R_over_N'] for i in ids] for s in ATT_SEEDS]).mean(axis=0)
        for baseline, names in [('MoE-LoRA', [f'MoE_{s}' for s in MOE_SEEDS]), ('Original H20', ['H20'])]:
            ref = np.array([[records[n][split][i]['R_over_N'] for i in ids] for n in names]).mean(axis=0)
            for group, domain_ids in [('all', list(groups)), ('structural', [d for d in groups if d.startswith('table2')]), ('large', [d for d in groups if d.startswith('large')])]:
                if not domain_ids:
                    continue
                delta, refmeans, attmeans, boot = [], [], [], []
                for d in domain_ids:
                    idx = np.array(groups[d])
                    vals = att[idx] - ref[idx]
                    delta.append(vals.mean()); refmeans.append(ref[idx].mean()); attmeans.append(att[idx].mean())
                    boot.append(vals[rng.integers(0, len(idx), (10000, len(idx)))].mean(axis=1))
                ci = np.quantile(np.mean(boot, axis=0), [.025,.975])
                comparisons.append({'split':split, 'group':group, 'baseline':baseline,
                    'att_macro':float(np.mean(attmeans)), 'baseline_macro':float(np.mean(refmeans)),
                    'delta_att_minus_baseline':float(np.mean(delta)),
                    'relative_improvement_pct':float(-np.mean(delta)/np.mean(refmeans)*100),
                    'paired_instance_bootstrap_ci95':ci.tolist(), 'replicates':10000})
    report['comparisons'] = comparisons
    report['uncertainty'] = 'Paired within-domain instance bootstrap, averaging fixed selected seeds first; does not estimate training-seed uncertainty.'
    save_json(OUT / 'summary.json', report)
    lines = ['# Two-seed preliminary development comparison', '',
        'H=20, CPU FP32 greedy. ATT-Joint: 2 completed seeds; MoE-LoRA: 3 released best adapters; original frozen H20. All evaluated on identical manifests. No final/test/Caserta scoring.', '',
        '| Split / group | ATT mean R/N | MoE mean R/N | H20 mean R/N | ATT vs MoE | ATT vs H20 |',
        '|---|---:|---:|---:|---:|---:|']
    pairs = defaultdict(dict)
    for c in comparisons:
        pairs[(c['split'],c['group'])][c['baseline']] = c
    for (split, group), cs in pairs.items():
        m, b = cs['MoE-LoRA'], cs['Original H20']
        lines.append(f"| {split} / {group} | {m['att_macro']:.6f} | {m['baseline_macro']:.6f} | {b['baseline_macro']:.6f} | {m['relative_improvement_pct']:+.2f}% | {b['relative_improvement_pct']:+.2f}% |")
    lines.extend(['', 'Positive percentages mean fewer relocations per container. All macro averages weight domains equally.', '',
        'Training curves reuse original validation records. Frozen baselines were freshly evaluated; horizontal lines show their reference score, not their training history.', '',
        'These are development data used for selection, not unbiased final estimates. Training coverage and budgets differ: ATT-Joint trains all 25 domains; MoE adapts the historical base on large-scale data. This comparison cannot isolate architecture or parameter-efficiency effects.', '',
        'Seed bands show min-max only. Bootstrap intervals condition on the selected models and cover instance uncertainty only.', '',
        '## Paired instance bootstrap: ATT minus baseline (negative is better)', ''])
    for c in comparisons:
        lo, hi = c['paired_instance_bootstrap_ci95']
        lines.append(f"- {c['split']} / {c['group']} vs {c['baseline']}: {c['delta_att_minus_baseline']:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}]")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    curves(report)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if '--plot-only' in sys.argv:
        report = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
        report['comparison_script_sha256'] = file_sha256(Path(__file__))
        curves(report)
        save_json(OUT/'summary.json', report)
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location('archived_moe_lora', LEGACY/'scrp/moe_lora.py')
    moe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = moe
    spec.loader.exec_module(moe)
    manifests = {s: load_episode_manifest(ROOT/f'data/online_crp/attention_multiscale_v1/{s}.json')
        for s in ('validation_seen','validation_unseen_geometry')}
    report = {'status':'PRELIMINARY_TWO_SEED_DEVELOPMENT', 'lookahead':20,
        'device':'cpu', 'precision':'fp32', 'torch_num_threads':2, 'torch_version':torch.__version__,
        'manifest_sha256':{s:file_sha256(ROOT/f'data/online_crp/attention_multiscale_v1/{s}.json') for s in manifests},
        'training_curve_sha256':{str(s):file_sha256(ROOT/f'output/attention_multiscale_v1/runs/a09_joint_seed{s}_n20_20261002/validation.jsonl') for s in ATT_SEEDS},
        'comparison_script_sha256':file_sha256(Path(__file__)),
        'archived_moe_implementation_sha256':file_sha256(LEGACY/'scrp/moe_lora.py'),
        'evaluations':{}, 'checkpoints':{}}
    specs = [(f'ATT_{s}', 'attention', s) for s in ATT_SEEDS] + [('H20','base',None)] + [(f'MoE_{s}','moe',s) for s in MOE_SEEDS]
    for name, kind, seed in specs:
        if kind == 'attention':
            run = ROOT/f'output/attention_multiscale_v1/runs/a09_joint_seed{seed}_n20_20261002'
            summary = json.loads((run/'summary.json').read_text())
            assert summary['status'] == 'BUDGET_COMPLETE' and summary['global_step'] == 6250
            policy, meta = load_attention_checkpoint(run/'best.pt',device='cpu')
            assert meta['global_step'] == summary['best_validation_step']
        else:
            policy, meta = load_legacy_v1_checkpoint(ROOT/'checkpoints/h20.best.pt',device='cpu')
            if kind == 'moe':
                path = LEGACY/f'output/moe_lora_long_20260917_seed{seed}/best.adapter.pt'
                payload = moe.load_moe_lora_adapter(policy,path,expected_base_checkpoint_sha256=meta['checkpoint_sha256'])
                meta = {**meta,'adapter_path':str(path),'adapter_sha256':file_sha256(path),
                    'adapter_metadata':payload.get('metadata',{}).get('completed_epoch')}
                policy.eval()
        report['checkpoints'][name] = meta
        report['evaluations'][name] = {}
        for split, episodes in manifests.items():
            path = OUT/f'{name}_{split}.jsonl'
            summary_path = OUT/f'{name}_{split}_summary.json'
            if path.exists() and summary_path.exists():
                result = json.loads(summary_path.read_text())
                assert result['checkpoint_sha256'] == (meta.get('adapter_sha256') or meta['checkpoint_sha256'])
            else:
                print(f'Evaluating {name}: {split}, {len(episodes)} instances', flush=True)
                result = evaluate_episodes(policy,episodes,lookahead_horizon=20,device='cpu',
                    method=name,model_seed=seed,checkpoint_sha256=meta.get('adapter_sha256') or meta['checkpoint_sha256'])
                rows = result.pop('rows')
                path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
                save_json(summary_path,result)
            if kind == 'attention' and split == 'validation_seen':
                assert abs(result['macro_mean_r_over_n'] - summary['best_validation_macro']) < 1e-12
                original_path = run/f"validation_records/step_{meta['global_step']:06d}.jsonl"
                original = {r['canonical_id']:r['R'] for r in read_jsonl(original_path)}
                refreshed = {r['canonical_id']:r['R'] for r in read_jsonl(path)}
                assert refreshed == original, 'Per-instance results differ from selected-checkpoint validation'
            report['evaluations'][name][split] = result
            print(f"  macro R/N={result['macro_mean_r_over_n']:.6f}, completed={result['completed_rate']}, seconds={result['evaluator_wall_seconds']:.1f}",flush=True)
        del policy
    summarize(report,manifests)
    print(json.dumps(report['comparisons'], indent=2),flush=True)
    print(f'Artifacts: {OUT}',flush=True)


if __name__ == '__main__':
    main()
