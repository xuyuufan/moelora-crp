"""Matched development evaluation of historical FULL and released MoE models."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.compare_a09_preliminary import read_jsonl, save_json, write_csv
from scrp.attention_data import file_sha256
from scrp.attention_evaluation import evaluate_episodes, load_legacy_v1_checkpoint
from scrp.online_crp import load_episode_manifest

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT/'output/attention_multiscale_v1/preliminary_two_seed_20261004'
OUT = ROOT/'output/attention_multiscale_v1/full_vs_moe_20261004'
ARCHIVE = ROOT.parent/'archive/20260916_cleanup/historical_outputs'
FULL_RUNS = {
    20260918: ARCHIVE/'h20_large_scale_lora_20260915_retry1/screen/seed20260918/FULL_recovery1',
    20260919: ARCHIVE/'h20_large_scale_lora_confirm_20260916/confirm/seed20260919/FULL',
}
MOE_NAMES = [f'MoE_{s}' for s in (20260913,20260923,20260933)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    prior = json.loads((PREVIOUS/'summary.json').read_text(encoding='utf-8'))
    manifests = {s:load_episode_manifest(ROOT/f'data/online_crp/attention_multiscale_v1/{s}.json')
        for s in ('validation_seen','validation_unseen_geometry')}
    manifest_hashes = {s:file_sha256(ROOT/f'data/online_crp/attention_multiscale_v1/{s}.json') for s in manifests}
    assert manifest_hashes == prior['manifest_sha256']
    rows_by_method = {}
    summaries = {}
    provenance = {'script_sha256':file_sha256(Path(__file__)), 'torch_version':torch.__version__,
        'device':'cpu','precision':'fp32','lookahead':20,'torch_num_threads':2,
        'manifest_sha256':manifest_hashes,'reused_report':str(PREVIOUS/'summary.json'),
        'reused_report_sha256':file_sha256(PREVIOUS/'summary.json'),'full_runs':{},'reused_files':{}}
    for name in ['H20']+MOE_NAMES:
        rows_by_method[name] = {}
        summaries[name] = prior['evaluations'][name]
        for split in manifests:
            path = PREVIOUS/f'{name}_{split}.jsonl'
            rows_by_method[name][split] = read_jsonl(path)
            provenance['reused_files'][str(path)] = file_sha256(path)
            assert all(r['precision']=='fp32' and r['hardware']=='CPU' and r['n']==20 for r in rows_by_method[name][split])
    for seed, run in FULL_RUNS.items():
        path = run/'best.pt'
        payload = torch.load(path,map_location='cpu',weights_only=False)
        assert payload['format']=='mijnrl_adaptation_model_v1' and payload['adaptation_mode']=='full'
        policy, base_meta = load_legacy_v1_checkpoint(ROOT/'checkpoints/h20.best.pt',device='cpu')
        assert payload['base_checkpoint_sha256']==base_meta['checkpoint_sha256']
        policy.load_state_dict(payload['model_state_dict'],strict=True)
        policy.eval()
        metadata = payload['metadata']
        training = read_jsonl(run/'metrics.jsonl')
        start = training[0]
        epochs = [r for r in training if r.get('record_type')=='epoch']
        complete = any(r.get('record_type')=='run_complete' for r in training)
        best = min(epochs,key=lambda r:r['validation_mean_relocations_per_container'])
        assert metadata['completed_epoch']==best['epoch']
        assert start['parameter_counts']['trainable_parameter_fraction']==1.0
        assert start['train_manifest_sha256']==file_sha256(ROOT/'data/online_crp/lora_domains/large_scale_n100_150/adapt_train.json')
        assert complete == (seed==20260918)
        provenance['full_runs'][str(seed)] = {'checkpoint':str(path),'checkpoint_sha256':file_sha256(path),
            'training_metrics':str(run/'metrics.jsonl'),'training_metrics_sha256':file_sha256(run/'metrics.jsonl'),
            'completed_epochs':max(r['epoch'] for r in epochs),'planned_epochs':start['adaptation_config']['epochs'],
            'run_complete':complete,'selected_epoch':metadata['completed_epoch'],
            'selected_validation_r_over_n':best['validation_mean_relocations_per_container'],
            'learning_rate':start['adaptation_config']['learning_rate'],
            'training_loss_mode':start['adaptation_config']['loss_mode'],
            'trainable_parameters':start['parameter_counts']['trainable_parameters'],
            'role':'primary_complete_seed' if complete else 'supplementary_incomplete_seed'}
        name = f'FULL_{seed}'
        rows_by_method[name] = {}
        summaries[name] = {}
        for split, episodes in manifests.items():
            rows_path, summary_path = OUT/f'{name}_{split}.jsonl', OUT/f'{name}_{split}_summary.json'
            if rows_path.exists() and summary_path.exists():
                result = json.loads(summary_path.read_text(encoding='utf-8'))
                rows = read_jsonl(rows_path)
                assert result['checkpoint_sha256']==file_sha256(path)
            else:
                print(f'Evaluating {name} / {split}: {len(episodes)} instances',flush=True)
                result = evaluate_episodes(policy,episodes,lookahead_horizon=20,device='cpu',
                    method=name,model_seed=seed,checkpoint_sha256=file_sha256(path))
                rows = result.pop('rows')
                rows_path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
                save_json(summary_path,result)
            rows_by_method[name][split] = rows
            summaries[name][split] = result
            print(f"  macro R/N={result['macro_mean_r_over_n']:.6f}, completion={result['completed_rate']}",flush=True)
        del policy
    comparisons = []
    score_rows = []
    rng = np.random.default_rng(20261101)
    for split, episodes in manifests.items():
        ids = [e.canonical_id for e in episodes]
        groups = defaultdict(list)
        for i,e in enumerate(episodes): groups[e.metadata['domain_id']].append(i)
        arrays = {}
        for name, splits in rows_by_method.items():
            rows = splits[split]
            assert len(rows)==len(ids) and {r['canonical_id'] for r in rows}==set(ids)
            assert len({r['canonical_id'] for r in rows})==len(rows)
            assert all(r['completed'] and r['illegal_actions']==0 for r in rows)
            mapping = {r['canonical_id']:r for r in rows}
            assert all(mapping[e.canonical_id]['N']==e.num_containers for e in episodes)
            arrays[name] = np.array([mapping[i]['R_over_N'] for i in ids])
        arrays['MoE_mean'] = np.mean([arrays[n] for n in MOE_NAMES],axis=0)
        for group, domain_ids in [('all',list(groups)),('structural',[d for d in groups if d.startswith('table2')]),('large',[d for d in groups if d.startswith('large')])]:
            if not domain_ids: continue
            values = {name:float(np.mean([arr[groups[d]].mean() for d in domain_ids])) for name,arr in arrays.items()}
            score_rows.append({'split':split,'group':group,**values})
            idx_all = np.concatenate([groups[d] for d in domain_ids])
            for full_name in ['FULL_20260918','FULL_20260919']:
                for ref_name in ['MoE_mean']+MOE_NAMES+['H20']:
                    diffs = arrays[full_name]-arrays[ref_name]
                    boot = np.mean([diffs[groups[d]][rng.integers(0,len(groups[d]),(10000,len(groups[d])))].mean(axis=1) for d in domain_ids],axis=0)
                    ci = np.quantile(boot,[.025,.975]).tolist()
                    comparisons.append({'split':split,'group':group,'full':full_name,'reference':ref_name,
                        'full_macro_r_over_n':values[full_name],'reference_macro_r_over_n':values[ref_name],
                        'delta_full_minus_reference':values[full_name]-values[ref_name],
                        'full_relative_degradation_pct':(values[full_name]/values[ref_name]-1)*100,
                        'paired_instance_ci95':ci,'replicates':10000,
                        'full_wins':int((diffs[idx_all]<-1e-12).sum()),'ties':int((abs(diffs[idx_all])<=1e-12).sum()),
                        'full_losses':int((diffs[idx_all]>1e-12).sum())})
    write_csv(OUT/'scores.csv',score_rows)
    save_json(OUT/'summary.json',{'status':'DEVELOPMENT_FIXED_CHECKPOINT_COMPARISON',
        'provenance':provenance,'evaluations':summaries,'scores':score_rows,'comparisons':comparisons,
        'limitations':['One complete FULL training seed; second FULL seed is incomplete and not averaged into primary result.',
            'FULL and MoE training budgets, learning rates, seeds and loss details differ.',
            'Bootstrap is conditional on these selected checkpoints; it estimates instance uncertainty, not training-seed uncertainty.',
            'No new final splits, historical sealed test, or Caserta scored.']})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'legend.frameon':False,'pdf.fonttype':42})
    fig,axes = plt.subplots(1,3,figsize=(12,4.2))
    panels = [('validation_seen','structural','21 structural domains'),('validation_seen','large','4 large domains (N=100-150)'),('validation_unseen_geometry','all','Unseen geometry validation')]
    names = ['H20','MoE_mean','FULL_20260918','FULL_20260919']
    labels = ['H20','MoE\n3 seeds','FULL\ncomplete','FULL\npartial*']
    for ax,(split,group,title) in zip(axes,panels):
        score = next(r for r in score_rows if r['split']==split and r['group']==group)
        colors = ['#666666','#D55E00','#0072B2','#CC79A7']
        x = np.arange(4)
        ax.scatter(x,[score[n] for n in names],s=65,c=colors,zorder=3)
        for i,n in enumerate(names):
            label_y = max(score[m] for m in MOE_NAMES) if n=='MoE_mean' else score[n]
            ax.annotate(f'{score[n]:.4f}',(i,label_y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        ax.vlines(1,min(score[n] for n in MOE_NAMES),max(score[n] for n in MOE_NAMES),color=colors[1],lw=3)
        ax.set_xticks(x,labels)
        ax.set_xlim(-.55,3.55)
        ax.margins(y=.3)
        ax.set(title=title,ylabel='Domain macro mean R/N (lower is better)')
    fig.suptitle('FULL vs MoE-LoRA | matched instances | H=20 | CPU FP32 greedy',fontsize=12)
    fig.text(.5,.015,'* Partial FULL: 5/10 epochs, supplementary only. MoE whisker: seed min-max, not a confidence interval.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    for ext in ['png','pdf']:fig.savefig(OUT/f'fig_full_vs_moe.{ext}',dpi=300)
    plt.close(fig)
    lines = ['# FULL 与 MoE-LoRA：固定检查点开发验证比较','',
        'H=20；同一批620个已见验证实例和100个未见geometry验证实例；CPU FP32 greedy；逐域等权平均R/N。', '',
        'FULL主结果为seed20260918的完整10-epoch训练，按旧验证集选择epoch1；seed20260919仅完成5/10 epochs，最佳epoch2，只作为补充。MoE为三份发布的best adapter。', '',
        '| 范围 | 原H20 | MoE三seed均值 | FULL完整seed | FULL中断seed（补充） |','|---|---:|---:|---:|---:|']
    for r in score_rows: lines.append(f"| {r['split']} / {r['group']} | {r['H20']:.6f} | {r['MoE_mean']:.6f} | {r['FULL_20260918']:.6f} | {r['FULL_20260919']:.6f} |")
    lines += ['', '## 完整FULL与MoE均值的逐实例配对比较','', '正差值表示FULL更差；区间仅估计实例不确定性，不能替代训练seed复核。','']
    for c in comparisons:
        if c['full']=='FULL_20260918' and c['reference']=='MoE_mean':
            lo,hi=c['paired_instance_ci95']
            lines.append(f"- {c['split']} / {c['group']}: FULL相对MoE的R/N变化{c['full_relative_degradation_pct']:+.2f}%；差值{c['delta_full_minus_reference']:+.6f}，95%区间[{lo:+.6f}, {hi:+.6f}]；FULL胜/平/负={c['full_wins']}/{c['ties']}/{c['full_losses']}。")
    lines += ['', '## 解释边界','',
        '这能比较现有检查点的效果。FULL与MoE的预算、学习率、训练seed和损失细节不同，不能把差距直接归因于专家路由，也不能据此认定全参数微调的性能上限。',
        '若要比较训练方法，需要从同一基座出发，以共同数据、预算和选模规则运行完整的多seed对照。',
        '未读取或评估最终test splits、旧密封测试或Caserta。旧MoE/H20结果复用前一轮逐实例记录，来源与哈希见summary.json；FULL结果为本次新评估。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(score_rows,indent=2),flush=True)
    print(f'Artifacts: {OUT}',flush=True)


if __name__=='__main__':main()
