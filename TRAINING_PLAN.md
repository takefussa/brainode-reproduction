# 学習開始前に提示する設定と現在の状態

2026-10-08 (Asia/Tokyo) 更新。ADNI1の468患者・1,405撮像のmesh生成と全件整合性確認を完了した。trainだけのPCA150をfitし、観測だけの初期BrainODE baseline `adni1_observed_v1` の100 epochsと最終test評価を2026-10-07 22:22 JSTに完了した。test60患者・175将来撮像で患者macro平均頂点距離0.3821仮定mm、初期元mesh維持0.4799仮定mm。連続c実験 `adni1_continuous_v1` のtrain-only pseudo36形状、CNN30 epochs、追加BrainODE20 epochs、Algorithm 2 N=10の全175 target評価を2026-10-08 01:00 JSTに完了した。全将来targetは0.3873仮定mm、最終撮像62 targetは0.4332（初期0.4258）で精度向上は確認できない。CN/AD撮像正解率87.45%、AUC0.9292だが、pseudo c再現MAE0.22948は一律0.5の0.22222を下回らない。全ADNI生成は完了16撮像を保存して停止した。[初期結果](artifacts/training_runs/adni1_observed_v1/RESULTS.md)、[連続c実験](artifacts/training_runs/adni1_continuous_v1/RESULTS.md)

先行実験の対象はADNI1の全CN/AD候補。患者splitは全ADNIのRID単位splitを継承し、train389患者・1,150撮像、test79患者・255撮像。timeはADNI1のtrainだけでfitした `(age−55.7288)/37.2931` を使用する。後述の全ADNI設定・台帳は後続比較のために保持する。[先行実験の台帳](artifacts/adni1_stage_baseline/summary.json)

基準実験はADNI1/2/3全体のCN/AD、年齢上下限なし、RIDで同一患者をまとめた患者単位80%/20%・seed42とする。PCAはtrain患者の全採用撮像でfitし、testを含めない。同日repeatは全て保持し、同時刻の別観測として扱う。AIBLとOASISは年齢・診断の対応を確認できた観測を追加実験で使い、確認できない記録は理由付きで保留する。OASIS2と3を混合せず個別の追加実験とする。元の全件台帳は保持し、欠損や不整合を推測で修正しない。

基準台帳は `artifacts/adni_all_baseline/` に作成済み。train1,164人・5,395撮像、test291人・1,351撮像。train min/maxは50.53/95.80歳で、timeは `(age−50.53)/45.27`。CN→AD患者はtrain18人、test6人。同日repeatを全保持し、同日age/診断の矛盾と後日のage逆転は0件。これはmetadataの準備であり、全meshの採用やPCA/model学習の完了ではない。[設定・確認結果](artifacts/adni_all_baseline/summary.json)

2026-10-07のmesh再点検では、最後のoptimizer更新が保存へ反映されない点を修正し、初回templateだけのサイズ初期化を比較した。2例の初回体積差は+12.44%/+30.33%から+0.18%/+0.14%、1患者4時点は絶対値0.3%未満・表面交差0になった。手元データ向けには明示flag `--template-initialization mask-volume` を推奨し、論文未記載の補完とする。全cohort QCは未完了。[実装監査と比較](artifacts/mesh_qc/initialization_review/REPORT.md)。以下の旧初回+12.81%等は調整前の履歴。

## 完了した準備と現在の作業

- ADNI1の全2402撮像を調べ、全ての年齢のCN/AD468人・1405撮像を候補台帳にした（CN712、AD693、年齢55.7288〜93.0219歳）。年齢による除外は行わない。MCI撮像を除き、途中にMCIがある患者のCN/ADは残す。
- ADNI1/2/3・AIBL・OASIS2/3の全15,727行を別の調査台帳へ保存した。文字列CN/AD・有限年齢は9,312撮像。ADNI全phaseでは重複をまとめて1,455人・6,746撮像、CN→AD端点を持つ24人を確認した。AIBLの年齢/臨床対応、OASISの診断対応は未確認で、最終の採用数ではない。公式資料にOASIS2/3の患者重複の可能性と個別使用の記載があるため、それぞれを追加する別実験を推奨する。[追加cohort調査](artifacts/multicohort_inventory/REPORT.md)
- 全1405撮像のparcellation headerを確認した。全て256³・uint16・spacing数値1/1/1・RAS・単位unknownで、affineは全て有限。voxel体積の数値は1との差が最大7.51e-8。これはheaderの確認であり、全maskのlabel/形状品質確認やTRSAAとのregistration確認が済んだという意味ではない。[header確認結果](artifacts/adni1_allage_eligibility/parcellation_headers_summary.json)
- train患者の最小・最大年齢からtimeを作り、その係数をtest/推論へ共通に使う処理を実装した。ADNI全体の患者splitとtrain係数fitは完了済み。
- 公式PointNetOptとtemplateを使ったmesh作成、PCA形状と連続cを使う論文の式(4)・Algorithm 1/2、3D CNNによる連続c推定を実装・検証した。ADNI1の全件mesh確認とPCA150 fitを完了し、観測だけのODE初期100 epochsと最終test評価を完了した。
- QC用の1患者・4時点を各5000iterationsでfittingした。Diceは0.898〜0.964。全1490頂点・2976faceの順序と、前meshを次の初期値にする処理は確認済み。
- **調整前に確認した問題（2026-10-06の履歴）**: 初回meshの体積がmaskより12.81%大きく、表面の交差が8組ある。後3時点の交差は0組。元maskの初回→最後の体積変化は−3.29%だが、meshは−12.84%で、fittingの誤差が時系列変化に混ざっている。10000iterationsでも体積差+11.46%・交差28組、公開lossそのままでも+14.51%・交差2組だった。各lossの作用だけでは原因を確定できず、単に反復を増やす/公開lossへ揃える変更は採用していない。2026-10-07に初期化を比較し、上記の改善を確認した。元の候補撮像は削除していない。

## 使うデータと形状表現

| 項目 | 現在の方針 | 状態・根拠 |
|---|---|---|
| データ | 先行実験はADNI1の年齢上下限なしのCN/AD。ADNI1/2/3全体、AIBL・OASIS2/3は後続比較として保持 | ADNI1は468患者・1,405撮像。追加候補全行の台帳は `artifacts/multicohort_inventory/` |
| MCI | その撮像だけ除外し、患者の他のCN/ADは保持 | 撮像単位の診断条件 |
| 単一撮像の患者 | 候補台帳から除かない | 未来との時間対がないのでODEの軌跡lossは作れない。PCA/c推定用の扱いは設定に従う |
| MRI・parcellation | MRI参照はTRSAA、形状は既存parcellationのlabel75 | 解析/学習のMRIはTRSAA |
| 空間単位 | headerの数値1を1mmとして扱う | 単位に関する仮定。元headerの単位はunknownで、変更しない |
| template | 公式の1490頂点・2976面、全時点で同じindex | 固定したface/頂点indexを使う |
| mesh作成 | 公開PointNetOpt、固定source、公開edge式、7 lossと論文重み。初回templateをmask体積で初期化し、以降は前meshを使用 | 初回サイズ調整は手元データ向け補完。2例と1患者4時点で確認済み。全cohort QC後に最終設定を説明 |
| train/test | 全ADNIの患者単位80%/20%・seed42 splitを継承 | ADNI1では389/79患者。phaseを跨ぐRIDは同じ側を保ち、ADNI1だけで再抽選しない。OASIS2/3は混合しない別実験 |
| 同日repeat | 全保持、同時刻の別観測として扱う | 採用設定。ADNI1では59組。OASIS2の1,368ファイルは373 MR sessionのrepeatを含む。未来予測は異なる撮像日・異なるtimeで評価 |
| PCA | 1490×3=4470座標を150次元へ。train患者の全1,150撮像でfull SVD/非whiteningのfitを完了し、testには同じPCAを適用 | 保持分散97.3792%、再構成平均頂点距離train0.1738/test0.2653仮定mm。公開コードの初回撮像のみのfitとの差を記録する |
| time | `(age−train_min_age)/(train_max_age−train_min_age)` | trainのみで係数fit・保存済み。testで再fit/clipしない |

## BrainODEの基本学習条件

| 項目 | 提示予定の設定 | 根拠・扱い |
|---|---|---|
| モデル | 公開 `model.py` の `ODEFuncWithAttention(150, 1)` | 構造は変更しない。入力はPCA150＋time1＋c1、Q/K/V256、FC512→150 |
| 積分 | RK4、各観測時刻を使用 | 論文・公開コード。連続cを更新するAlgorithm 2の区間数は別途確定 |
| optimizer | AdamW、lr=0.0005 | 論文A.4。公開コードの既定betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01を使用する案 |
| epochs | 初期100、追加20 | 初期は論文A.4。公開train.pyの既定1000とは異なる。連続c実験の追加20は補完で、合計120を論文100と同一視しない |
| lr schedule | StepLR、20 epochsごとに0.5倍を使用する案 | 公開コードの `epoch//5` を100 epochsへ適用。論文A.4にはscheduleの記載なし |
| batch・seed | 2患者分のlossで1回更新、seed42を使用する案 | 公開コードの設定。各患者は独立に積分し、最後の端数も更新する。公開コードのgradient取りこぼしを修正した関数を使用 |
| loss | PCA空間の前向き・後向きの全時間対でL2二乗距離、時間対数で平均 | 論文Algorithm 1。reductionの具体値は補完として記録。sourceのcを積分中に固定し、未来診断を入力しない |
| augmentation | 実頂点を共通原点から一様なfactor∈[0.98,1.02]でscale | 論文A.4。PCAの中心offsetも処理する。公開コードのGaussian noiseは追加しない |
| checkpoint選択 | 最終epochを主結果に使う案 | test結果で最良epochを選ばない。validationやearly stoppingを追加するなら別途明示 |

## 連続cと評価で開始前に確定すること

観測だけの初期baselineは上表の100 epochs・AdamW5e-4・StepLR20/0.5・2患者batch・seed42・最終epochを固定して実行する。trainの時間対あり299患者を使い、全1,150train撮像をPCAデータに保持する。testの時間対あり60患者・175将来撮像を最初の観測から予測し、患者ごとの平均をさらに等重みで平均する値を主集約とする。将来診断を入力せず、testでcheckpointを選ばない。[固定config](artifacts/training_runs/adni1_observed_v1/config.json)

PCA150再構成で628撮像に自己交差がある。影響面積割合はmedian0.0734%、最大3.8484%。初期baselineは候補を除外せず、元mesh targetへの座標予測誤差と予測表面の交差を記録する。CNNのvoxel化・体積の妥当性は別途確認する。[PCA表面geometry](artifacts/model_data/adni1_pca150_v1/geometry/REPORT.md)。public modelは患者独立のstate batch=1でattention softmax=1となる呼出しも維持する。

観測教師値はCN=0、AD=1。式(4)のpseudo cは患者のCN→AD間隔中の相対位置で、年齢正規化の共通係数は比で相殺される。pseudo形状は学習済みODEの前向き/後向き予測の平均から作る。trainの患者・端点だけを使い、test端点から教師データを作らない。

推定器はvoxel入力3D CNN（Conv16/32/64、ReLU/MaxPool、global average pooling、sigmoid連続出力）とsoft-label BCEを使う補完仕様。voxel gridは後述の条件で全件準備済み。全4 train端点組に各9内点、計36pseudoを生成し、全1,150実測と各例1回/epochでCNN30 epochs・batch16・AdamW5e-4・StepLR10/0.5を学習する。追加ODEは観測100 epochsのcheckpointから20 epochs・batch2患者・AdamW5e-4・StepLR20/0.5、converter患者内で実測/pseudo軌跡lossを1/2ずつ混合する。Algorithm 2はN=10を事前固定する。これらの学習密度・配分・CNN最適化・Nは原著未記載の補完であり、合計120 ODE epochsを原著100 epochsと同一視しない。先行ADNI1のCN→ADはtrain2患者・test0患者で、連続c生成とconversion予測の検証範囲は限定される。全ADNIをつなぐとtrain18人・test6人。追加候補の全行を保持し、未確認の診断や年齢を推測で修正して学習へ入れない。

voxelはtrainのみでfitしたorigin=(-31,-34,-33)、128³、pitch0.5仮定mm、固定voxel中心のray parity判定（3 rays）。全撮像の3/7 rays比較とmesh/voxel体積差を記録する。診断22例では旧surface-fillの約20%体積膨張がcenter-occupancyで−0.168〜+0.140%になった。既知の自己交差を修復する処理ではない。CNNの学習条件・pseudo生成・追加ODEとAlgorithm 2は前述の固定条件で完了した。[voxel診断](artifacts/model_data/adni1_pca150_v1/voxel_center_diagnostic/REPORT.md)、[連続c固定条件](artifacts/training_runs/adni1_continuous_v1/README.md)

初期baselineの保存済み175予測も固定gridでvoxel化できた。失敗0、3/7 raysの判定差0、voxel体積と符号付きmesh体積の差は最大0.3674%。自己交差102形状は残存する。連続c実験のpseudo生成、CNN学習、追加ODE、Algorithm 2再評価も完了し、自己交差103/175、全1,750中間入力で3/7 rays差0、体積差最大0.3855%。数値整合性だけで正しい解剖学的内部やCNN出力への影響がないことを示さない。自己交差を防ぐ制約を追加する場合は、公開モデルからの変更点と座標誤差・表面品質の両方を比較する。[初期予測voxel確認](artifacts/training_runs/adni1_observed_v1/predicted_voxel_review/REPORT.md)、[連続c結果](artifacts/training_runs/adni1_continuous_v1/RESULTS.md)

主評価は予測meshと元の観測meshの対応頂点間の平均Euclidean距離（仮定mm）。PCAで再構成したtargetだけに対する誤差を主結果にしない。`evaluation.py` に元meshをtargetとする誤差、PCA再構成誤差、公開コードのPCA targetへの座標MAE、初期形状をそのまま将来へ置く予測の誤差を分けて計算する処理を実装した。解析的な例で「PCA targetへの誤差0でも元meshへの誤差が残る」ことを検証し、整数座標のoverflowや不正な形状の扱いも含め4 testsが通過した（`audit/evaluation_tests.txt`）。1-shotの未来予測を最初の評価とし、4-shot評価を追加する場合は対象時刻・対象数を事前に説明する。観測mesh自体のfitting誤差も別に示す。評価の集約方法・予測時点は最終設定に明記する。

根拠: [BrainODE論文A.1/A.3/A.4・式(4)・Algorithm 1/2](https://proceedings.neurips.cc/paper_files/paper/2025/file/abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf)、[取得済み公開モデル](/home/take/brainode/model.py)、[取得済み公開train.py](/home/take/brainode/train.py)、[現在のSPEC](/home/take/brainode/SPEC.md)、[mesh QC結果](/home/take/brainode/artifacts/mesh_qc/REPORT.md)。

## 一括mesh生成の準備

recipe 3の一括処理を実装した。同日repeatは日付・UID数値順で全保持し、直前に処理したmeshを引き継ぐ。同日内順序は補完として固定する。コード・入力台帳をsnapshotに固定し、設定・環境・source/artifact hashを照合して途中から再開できる。関連18 testsが通過した。

実データ5患者・12撮像のQCで、年齢端、単一撮像、CN/AD repeat、CN→AD、ADNI phase間を確認した。体積差−1.60%〜+1.34%、Dice0.921〜0.962、表面交差・品質flag0件。[QC報告](artifacts/mesh_qc/cohort_pilot_v3/review/REPORT.md)全6,746撮像の台帳をこの検証例へ絞る処理は行わない。品質flagはレビューの対象を示し、撮像の自動除外には使わない。このpilotはmesh fittingとQCを対象とする。PCA/BrainODEの先行学習結果は冒頭のADNI1結果に記録する。

全1,455患者・6,746撮像の生成は完了16撮像を保存して停止した。ADNI1の468患者・1,405撮像はGPU4・5の2枚・各4 workerで生成し、2026-10-07 19:30 JSTに完了した。全件hash・topology・引継ぎと品質flag73撮像の詳細を確認した。元mask分離28撮像の副成分は1〜16voxel、自己交差42撮像の影響面積は0.0094〜0.5126%、Dice<0.90は7撮像。異なる台帳のmesh cacheは混用しない。[最終確認](artifacts/mesh_runs/adni1_stage_v3/review/REPORT.md)、[flag詳細](artifacts/mesh_runs/adni1_stage_v3/flag_review/REPORT.md)。初期ODE baselineの実行GPUは4、使用GPU上限は4・5の2枚。
