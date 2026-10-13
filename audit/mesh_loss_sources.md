# mesh fitting: 論文と公開コードの差

2026-10-06。一次資料と実装を比較し、補完設定を実装した。固定source＋公開edgeで実データ比較と4時点QCを行った。現在の設定と比較結果は以下および [QC報告](/home/take/brainode/artifacts/mesh_qc/REPORT.md) に記録する。

## 一次資料

- [BrainODE A.1・式(5)・Figure 4](https://proceedings.neurips.cc/paper_files/paper/2025/file/abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf)。手元の32ページPDFを参照。
- [Park et al., NeuroImage 310:121145 (2025), §2.3](https://www.pure.ed.ac.uk/ws/portalfiles/portal/498397382/1-s2.0-S1053811925001478-main.pdf)。BrainODE A.1の参考文献[11]と一致する出版版。`references/Park-NeuroImage2025-121145.pdf` に保存し、SHA256は `202fafce455ae22cd5c96a45dd1aae9624f97074e335e0ccfeaf172151e75ef1`。対応するtxtも保存済み。
- 取得済み公式BrainSubcorticalStructureの `models/loss.py`, `models/loss_3d.py`, `optimize_p2p.py`。commit `d65ffe5da55e53e1431cc93f91e3b348497b8e5d`。

## 確認できた差

| 項目 | 論文の記述 | 公開コード | 再現上の扱い |
|---|---|---|---|
| normal displacement | BrainODEはL2の項と重み1を記載。Park出版版はtemplateのnormalからの変化を抑えると説明 | normal用importがコメントアウトされ、項と実装ファイルがない | 方向の変化を抑える項が必要。「normal方向への頂点変位」とは確定しない。vertex/face normalの別、平均やsum等のreductionは本文だけで未確定 |
| vertex displacement | BrainODEはL2表記。Parkは移動距離のroot mean squareと説明 | 最初のsourceと現在のverticesの座標MSE | 同一の式とは扱わない。norm/reductionを実装するときに明示する |
| edge loss | 両論文はedge lengthのvariance | length差の二乗をedge数で割った値の標準偏差。単一mesh・target_length=0なら `std(length²/E)` | 通常の `var(length)` とは別。重み1500は同じでも数値スケールが変わるため説明が必要 |
| 同一scan内の更新 | BrainODE Figure 4は前iterationのmeshから変形。Parkは変位を反復的に蓄積すると説明 | 各iterationで固定した `pred_mesh` のverticesを再取得。変形済みmeshを次iterationのsourceへ戻していない | SGD iterationごとの変位加算と解釈した補完実装は実データで破綻した。図だけではoptimizer・graphの更新手順まで確定できず、固定sourceと比較して採用方針を相談する |
| longitudinal処理 | BrainODEは前時点の変形済みmeshを次時点の初期値にする | 公開optimizerは単一scan向け | 全採用撮像の時系列処理が必要。2例のQC端点だけを採用する意味ではない |

BrainODEの重み `{cf, pm, vert, norm, edge, cons, lap} = {0.5, 3, 1, 1, 1500, 1, 5}` と共有topologyは維持する。lossの名称が一致するだけで数式・reductionまで確定したことにはしない。

## 実装した補完設定

- vertex項: 撮像の初期meshからの頂点移動距離のRMS。norm項: 同じ初期meshからのunit vertex normal変化のRMS。vertex normalはPyTorch3Dのarea-weighted定義。双方の数値的な原点を平滑化するepsilonは1e-12で、原点のpenaltyは0にする。
- edge項: 最初はedge長さのpopulation varianceで比較し、最新の採用方式では公開コードの `std(length²/E)` を使う。これを論文のvarianceとは呼ばない。重み1500は維持する。
- Chamfer: 公開コードに合わせmesh表面からboundary点数と同数をsample。targetのboundary点自体は間引かない。BrainODE A.1のViという表記だけで頂点のみのChamferと確定せず、引用元Park §2.3のsurface samplingを根拠にする。
- point-mesh: 公開コードと同じPyTorch3D `point_mesh_face_distance`。point→faceとface→pointの双方向を含む。引用元Park §2.3は双方向と明記し、BrainODE A.1の簡略なpoint-to-face表記より実装の内容が具体的である。
- 各iteration: 現在の採用方式は公開コードと同じ固定sourceに予測変位を足す。図を逐次加算と解釈した方式はつぶれた比較履歴として保持する。公開PointNetOpt自体のarchitectureは変更しない。
- 各撮像: 直前の撮像の変形済みmeshを初期値にし、その撮像の初期値をregularizerの参照とする。faceとvertex順序は一切作り直さない。
- AdamW lr=5e-4、5000iterations、1000ごとにlr半減は引用元と公開optimizerから補完。動作確認の短いiterations指定は本設定とは分けて保存する。

これらは著者の未公開コードの厳密な復元ではない。unit/reductionの選択による数値スケールを含め、品質確認前のmeshをPCA/BrainODE学習に採用しない。

## 実データで確認できた影響

同一撮像・seed・5000iterationsで、逐次加算の解釈＋辺長varianceはDice0.000、固定source＋辺長varianceは0.773、固定source＋公開edge式は0.898だった。7項と重みは共通。再現を急いで逐次加算の補完をそのまま採用すると無効なmeshを作る。固定source＋公開edge式を採用し、論文のvariance記述との差を記録した。論文のnormal項は引き続き省略しない。詳細・画像・定義は [実データQC報告](/home/take/brainode/artifacts/mesh_qc/REPORT.md) を参照する。

追加QCでは、同じ初回撮像で10000iterationsにしても体積差+11.46%・表面交差28組が残った。公開criterionを変更せず読み込み、公開vertex座標MSEとnormal省略まで揃えた5000iterations比較も行い、Dice0.891・体積差+14.51%・交差2組だった。この追加比較は未採用。公開lossだけで初回品質が解決したと扱わず、normalの省略とvertex reductionの変更の個別効果も分離したとは主張しない。

## 引用元の設定を自動で移植しない

Park §2.3にはICVの立方根による点群のscale調整、4002頂点、3000点、5000iterations、1000iterationごとのlearning rate半減が記載されている。BrainODE用には1490頂点の公式templateを使う。BrainODE本文で確認できないICV補正・sampling数等を、引用元にあるだけで無断で追加しない。元論文のmanual mask向けmorphology処理も今回のlabel75へ自動適用しない。
