# 年齢のtime入力と連続cの区別

2026-10-07更新。全CN/AD年齢を対象にし、trainの最小・最大年齢で共通のtime正規化を行う。

## 今回のデータへ適用する定義

`t=(age−a_min)/(a_max−a_min)`。`a_min`・`a_max` は、採用された全train患者のCN/AD撮像の年齢から求める。患者ごとに正規化せず、test/将来予測にも同じ係数を使う。年齢差は `Δt=Δage/(a_max−a_min)` なので、どの患者でも1年の尺度が同じになる。train範囲外のtest/将来の年齢は0未満・1超のtimeになり得るが、clipしない。

先行実験はADNI1の全CN/AD **468人・1405撮像**。年齢は55.7288〜93.0219歳。MCI撮像は除外し、途中にMCIがある患者のCN/ADは残す。全ADNIのRID単位splitを継承し、ADNI1はtrain389人・1,150撮像/test79人・255撮像。ADNI1のtrainだけでfitした年齢範囲は55.7288〜93.0219歳、係数は `(age−55.7288)/37.2931`。候補全体の範囲と数値が一致するのは、両端の患者がtrain側にいるためであり、testを係数fitへ使っていない。[先行実験の係数](../artifacts/adni1_stage_baseline/time_scale.json)

後続の全ADNI比較用splitはtrain1,164人/test291人で、train年齢50.53〜95.80歳から `(age−50.53)/45.27` をfit済み。それぞれの実験のtrain係数を保存し、testでは再fitしない。[全ADNIの係数とsplit](../artifacts/adni_all_baseline/summary.json)

`lab_dataset.py` の `AgeTimeScale.fit(..., train_subjects=...)` と `normalized_visits` に実装した。CLIに `--train-subjects`（1行1患者IDの一覧）を渡すと、time列と、係数・fit対象の患者/撮像を記録した `time_scale.json` を出力する。指定がない候補台帳には仮のtime列を作らない。現在のADNI1台帳は `artifacts/adni1_stage_baseline/` に保存し、`audit/prepare_adni1_stage.py` でsplitを継承して係数を作成した。旧台帳は比較履歴として保持する。

この変更は手元データに合わせた同じmin-max正規化の適用である。論文の固定値65・30を今回の対象年齢へ合わせて変更する適用設定である。

## 論文・公開コードとの照合

論文は対象を65〜95歳とし、`t=(age−65)/(95−65)=(age−65)/30` と固定する。65は選択した年齢範囲の下限、30はその幅の30年間。実際の標本にちょうど65歳や95歳がいなくても使える。

| 年齢 | 論文のtime |
|---:|---:|
| 65 | 0 |
| 70 | 1/6 |
| 80 | 1/2 |
| 95 | 1 |

年齢差Δageに対して時間差は `Δt=Δage/30`。1年はどの患者でも1/30、3年は0.1になる。

公開 `dataset.py` と `train.py` はpickleに保存済みのtimeを読み、ODEへ渡す。年齢からtimeを作る処理は公開されていないため、コードだけから「著者は手元のmin/maxに合わせた」とは判断できない。

`paper_cognition.normalize_age` の65・30の式は論文の照合用関数として残し、今回の実dataset処理では呼ばない。対象年齢の選択とtimeの変換は別の設定であり、現行の全CN/AD年齢とtrain min/max正規化を適用する。

## 連続c

連続cは別の量である。NCからADへの観測端点について、式(4)は `c_tilde=(sample_time−NC_time)/(AD_time−NC_time)` と、その患者の実際の時間差を分母に使う。NC80歳・AD84歳なら82歳はc=0.5。timeを全員共通のaffine変換で正規化してもこの比では係数が相殺され、同じ値になる。

根拠: [BrainODE論文A.4・式(4)](https://proceedings.neurips.cc/paper_files/paper/2025/file/abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf)、[取得済み公開train.py](/home/take/brainode/train.py:83)、[最新候補台帳](/home/take/brainode/artifacts/adni1_stage_baseline/summary.json)、[実装と検証](/home/take/brainode/tests/test_lab_dataset.py)。
