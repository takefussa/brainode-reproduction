"""Supplement the fixed baseline with a latest-visit-only paper comparison."""

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
REPORT=PROJECT/'artifacts/training_runs/adni1_observed_v1'
PAPER='https://proceedings.neurips.cc/paper_files/paper/2025/file/abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf'


def main():
    with (REPORT/'RESULTS.csv').open() as file:
        rows=list(csv.DictReader(file))
    groups=defaultdict(list)
    for row in rows:
        groups[row['pid']].append(row)
    selected=[];means=[]
    keys=['prediction_mean_vertex_euclidean','static_source_mean_vertex_euclidean',
        'pca_static_mean_vertex_euclidean','pca_target_mean_vertex_euclidean',
        'public_coordinate_mae_to_pca_target','pca_reconstruction_mean_vertex_euclidean','horizon_years']
    for pid,group in groups.items():
        day=max(r['target_day'] for r in group)
        latest=[r for r in group if r['target_day']==day]
        selected.extend(latest)
        means.append([np.mean([float(r[k]) for r in latest]) for k in keys])
    values=np.asarray(means)
    if len(groups)!=60 or len(selected)!=62:
        raise ValueError('unexpected coverage of latest-date observations')
    summary=dict(task='first observation to latest observed scan date; retain every same-day repeat',
        observations_removed_from_dataset=0,training_or_prediction_rerun=False,
        test_subjects=len(groups),latest_date_targets=len(selected),
        subject_macro_metrics=dict(zip(keys,values.mean(0).tolist())),
        relative_reduction_vs_raw_static=float(1-values[:,0].mean()/values[:,1].mean()),
        relative_reduction_vs_pca_static=float(1-values[:,0].mean()/values[:,2].mean()),
        patients_improved_vs_raw_static=int(sum(values[:,0]<values[:,1])),
        horizon_years=dict(mean=float(values[:,-1].mean()),median=float(np.median(values[:,-1])),
                          min=float(values[:,-1].min()),max=float(values[:,-1].max())),
        latest_targets_with_self_intersection=sum(int(r['prediction_self_intersection_pairs'])>0 for r in selected),
        units='one numerical source unit assumed to be one mm',
        paper_reference=dict(url=PAPER,closest_task='Table 2 irregular 1-shot hippocampus',
            mean_euclidean_error_mm=.492,test_subjects=79,
            cohorts=['ADNI','AIBL','OASIS2','OASIS3'],
            regular_1shot_mm={'LBC1936':.365,'AIBL':.406}),
        comparison_status='same nominal task and error scale, different data/preprocessing and incomplete continuous-c pipeline; no claim of outperforming the paper',
        public_metric_note='train.py computes coordinate MAE to PCA-reconstructed target; paper describes mean Euclidean distance; these are distinct metrics')
    (REPORT/'PAPER_COMPARISON.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (REPORT/'latest_visit_metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    metrics=summary['subject_macro_metrics']
    lines=['# BrainODE論文との精度比較','',
        '初期100 epochsの最終checkpointと保存済み予測を固定し、論文§4の1-shotに近い「最初から最終撮像日」の集約を追加した。元の全対象・予測・主評価は変更しない。同日repeatは全保持して患者内で平均する。','',
        '| 結果 | 対象 | 平均頂点Euclidean距離（mm） |','|---|---|---:|',
        f'| 現行主評価 | ADNI1、60患者・全175将来撮像 | 0.3821 |',
        f"| 最終撮像日だけの補足評価 | ADNI1、60患者・62撮像 | {metrics['prediction_mean_vertex_euclidean']:.4f} |",
        f"| 初期元meshを維持 | 同じ最終撮像日target | {metrics['static_source_mean_vertex_euclidean']:.4f} |",
        '| 論文Table 2、1-shot海馬 | ADNI・AIBL・OASIS2/3混合、79患者 | 0.492 |','',
        f"現行最終撮像日評価では静的予測から{summary['relative_reduction_vs_raw_static']*100:.2f}%改善、{summary['patients_improved_vs_raw_static']}/60患者で改善。予測間隔は平均{summary['horizon_years']['mean']:.3f}年、範囲{summary['horizon_years']['min']:.3f}〜{summary['horizon_years']['max']:.3f}年。",'',
        '論文Table 1の定間隔1-shot海馬はLBC1936が0.365、AIBLが0.406mm。別cohort・別時間間隔なので横並びの順位付けには使わない。', '',
        '直接の優劣を判断できない理由は、cohort・split・年齢/time・parcellation/mesh/PCAの条件が異なり、こちらはpseudo連続c/CNN/Algorithm 2を未使用の初期baselineである点。論文の不規則1-shot testに含まれる79患者の個別予測間隔は公開表から再構成できない。', '',
        f"最終撮像日targetの{summary['latest_targets_with_self_intersection']}/{len(selected)}予測に自己交差がある。主評価全175予測では102件、交差に関係する面積割合はmedian0.130%、最大1.919%。数値誤差の比較と表面品質を分ける。",'',
        '公開train.pyのmm_lossはPCA再構成targetへのXYZ座標MAE。一方、論文は平均Euclidean距離と記述する。今回の最終撮像日について前者は0.1596、PCA targetへの平均Euclidean距離は0.3203、元targetへの主距離は0.4258仮定mmであり、別の値である。0.1596を論文0.492と比較しない。','',
        '距離は元headerの空間単位unknownに対し、数値1を1mmとする既存仮定。表面対応indexを固定した比較で、予測したMRI voxelや診断正解率の評価ではない。', '',
        f'[論文Table 1/2・§4]({PAPER}#page=7)、[論文Table 6・Appendix B]({PAPER}#page=24)、[公開codeのMAE](/home/take/brainode/train.py:232)、[補足数値](PAPER_COMPARISON.json)、[最終撮像日の全結果](latest_visit_metrics.csv)。','']
    (REPORT/'PAPER_COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
