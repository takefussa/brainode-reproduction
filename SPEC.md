# BrainODE 再現実装の仕様

更新日: 2026-10-08 (Asia/Tokyo)。現行設定、根拠、検証範囲を記録する。

## 目的と保存先

彌冨研究室のlongitudinalデータで、左海馬の対応付きmesh、PCA150次元、BrainODE、連続c推定、未来形状の数値評価と3D表示まで実装する。BrainODE論文と著者公開実装を再現基準とし、両者の差と未記載の補完を記録する。

コード・文書・軽い結果は `/home/take/brainode`、大きい環境・cache・全cohort生成物・checkpointは `/data5/take/brainode` に置く。研究元データ `/data2/radiology_datas/clean3/` は読み取り専用。画像・header・metadata・既存scriptを変更せず、作業物の削除は本projectの対象に限定する。

不要なregistration、resampling、再segmentation、smoothing、ICV補正、独自architectureを追加しない。通常の実装修正は根拠と検証を記録する。重大なデータ条件の変更や再現を妨げる不足は、根拠と影響を整理して相談する。

## データと時間

現在の先行実験はADNI1の468患者・1,405撮像。全ADNIのRID単位splitを継承し、train389患者・1,150撮像、test79患者・255撮像とする。ADNI1のtrainだけでtime係数をfitし、`t=(age−55.7288)/37.2931` を使用する。同日repeatと単一撮像患者を保持する。全ADNIの台帳とsplit/timeは後続比較用として保持し、全ADNI生成は停止している。[先行実験の設定](artifacts/adni1_stage_baseline/summary.json)

基準実験はADNI1/2/3の全CN/AD年齢。撮像単位で診断・有限年齢を確認し、MCI撮像だけを除外する。途中にMCIがある患者のCN/AD、単一撮像患者、同日repeatを保持する。全元行と除外理由を台帳へ保存し、診断や欠損年齢を推測で補完しない。

ADNIはRIDでphaseを跨ぐ同一患者をまとめる。患者単位seed42・80%/20%のsplitを作成済み。1,455人・6,746撮像のうちtrainは1,164人・5,395撮像、testは291人・1,351撮像。同日repeatは2,260グループ。CN→AD患者はtrain18人、test6人。同日の年齢/診断不一致と後日の年齢逆転は0件。[基準台帳と設定](artifacts/adni_all_baseline/summary.json)

timeは `t=(age−train_min_age)/(train_max_age−train_min_age)`。全ADNI比較のtrain係数は50.53〜95.80歳による `(age−50.53)/45.27`、先行ADNI1実験は上記の係数を使う。各実験のtest/推論へそのtrain係数を適用し、再fit・患者ごとの正規化・clipを行わない。metadataの年齢を使い、初回年齢と日付差から作り直さない。論文A.4の65〜95歳・分母30との差は対象年齢範囲に合わせた適用設定として記録する。[timeの定義](audit/time_definition.md)

6cohort全15,727 metadata行の調査台帳を保持する。文字列CN/AD・有限年齢は9,312撮像。AIBLの年齢/臨床対応、OASISの診断対応と欠損は未確認を含む。対応を確認できた観測を追加比較に用いる。OASIS2/3は患者重複の可能性から個別の追加実験とし、混合した無作為splitを作らない。[追加cohort調査](artifacts/multicohort_inventory/REPORT.md)

## MRIとparcellationの座標

解析MRIはTRSAA。meshは既存parcellationのlabel75から作る。75=Hippo_Lは手元の `meta/info/Level5.txt` と `Multilevel_stripped.txt` で確認済み。公開元のFreeSurfer系label17をそのまま手元データに適用しない。

parcellationとTRSAAは異なる配列・affineを持つ。ADNI1の2例ではparcellationとunifiedが同じ格子・affineだった。unifiedは座標確認用の参照に限定し、解析/学習MRIへ採用しない。native-to-TRSAAの対応変換は未確認で、headerだけによる投影はregistrationではない。配列のresizeで正しい対応を作れるとは扱わない。[座標QC](audit/hippocampus_pair_qc/REPORT.md)、[全画像格子と脳領域の比較](artifacts/image_geometry_review/REPORT.md)

headerの空間単位はunknown。spacing数値1を1mmとする仮定を出力に記録し、元headerは変更しない。BrainODE本体の入力はMRI配列ではなくmeshのPCA係数。native parcellationからのmesh fittingとTRSAAへの重ね合わせは別工程である。

## 対応付きmesh

templateは著者公開の左海馬1490頂点・2976面。BrainODE付属templateとBrainSubcorticalStructureの `17_temp.obj` は全頂点・面が完全一致する。全時点で同じ頂点indexとface topologyを保持し、独立marching-cubes meshをBrainODEへ直接入力しない。

境界点は公開処理と同じ2³近傍convolutionで抽出し、parcellationのaffineで座標へ変換する。中心合わせと拡縮を含まないrigid ICPでsourceに合わせ、公開PointNetOptを変更せず使用する。撮像内は固定sourceから変位を計算し、次の撮像は前のfitted meshを初期値とする。

PointNetOptは各撮像で新規に初期化し、その撮像のparcellation境界へ合うよう5000 optimizer更新する。事前学習済みweightや前撮像のnetwork weightは読み込まず、前撮像から引き継ぐのはfitted meshの頂点座標と共通faceだけ。境界抽出・rigid ICPは固定した幾何処理、meshの変形はニューラルネットの撮像別最適化である。これは患者間で共通の未来予測を学ぶBrainODE/連続c推定器のdataset学習とは別工程である。

7項の重みはchamfer0.5、point-mesh3、vertex1、normal1、edge1500、normal consistency1、Laplacian5。edgeは公開式 `std(length²/E, unbiased=True)` で、論文の辺長varianceとは異なる。vertex/normalの詳細は撮像初期meshを参照したRMS、area-weighted vertex normal、epsilon1e-12で補完する。[loss資料比較](audit/mesh_loss_sources.md)

初回templateだけを `factor=(mask_volume/template_volume)^(1/3)` で初期化する補完を実装した。mask・MRI・最終meshの体積を強制的に揃える処理ではない。後の撮像へ追加scaleを行わず、前meshを引き継ぐ。初期化は `--template-initialization mask-volume`、公開元サイズの比較は `unscaled` として明示指定する。これは論文/公開codeに明記されていない補完である。

最後のoptimizer更新後のparameterから出力meshを再計算する。recipe version=3とsource/code/template/face/artifact hashを記録し、旧recipeや異なる初期化のcacheを再利用しない。loss CSVは各更新前の値である。

2例の初回5000更新比較では体積差が+12.44%/+30.33%から+0.18%/+0.14%、Diceが0.901/0.797から0.946/0.932へ改善し、表面交差は両例0になった。1患者4時点は体積差の絶対値0.3%未満・交差0。関連14 testsが通過した。全cohort品質や厳密な解剖学的頂点対応は未検証。[監査と実データ比較](artifacts/mesh_qc/initialization_review/REPORT.md)

## PCAとBrainODE

1490×3=4470座標を150次元へ変換する。ADNI1のtrain患者の全1,150撮像だけでPCA150をfitした。testの255撮像にも同じ基底を適用する。full SVD、whiten=Falseで保持分散97.3792%、平均再構成頂点距離はtrain0.1738、test0.2653仮定mm。公開codeの初回撮像のみのfitとの差を記録する。[PCAと全撮像台帳](artifacts/model_data/adni1_pca150_v1/REPORT.md)

PCA150再構成は628撮像で表面交差を生じた。交差する面の表面積割合はmedian0.0734%、最大3.8484%。PCAは非交差曲面を保証しないため、座標予測baselineとCNN voxel化・体積妥当性の確認を分ける。元mesh・基底の修復や候補の除外は行わない。[再構成geometry](artifacts/model_data/adni1_pca150_v1/geometry/REPORT.md)

モデルは公開 `ODEFuncWithAttention(150, 1)`。入力はPCA150+c1+time1、Q/K/V256、FC512→150。RK4で実観測時刻を積分する。`paper_training.py` はAlgorithm 1の前向き・後向き全時間対とL2二乗距離を実装し、sourceのcを区間内で固定する。未来診断を入力しない。lossは時間対数で平均する補完設定で、batch末尾のgradientも更新する。

augmentationは共通原点から実頂点をfactor∈[0.98,1.02]でscaleし、PCA中心offsetも処理する。公開codeのGaussian noiseは追加しない。観測だけの初期baseline `adni1_observed_v1` はAdamW5e-4、100 epochs、2患者batch、seed42、StepLR20 epochsごと0.5、最終epochで評価する固定設定で2026-10-07 22:22 JSTに完了した。train389患者のうち時間対を持つ299患者・3,412順序付き時間対で学習し、時間対のない90患者もPCA用データに保持する。[固定設定と結果](artifacts/training_runs/adni1_observed_v1/README.md)

患者ごとの独立積分は公開train.pyと同じ。公開modelのattentionはstate batch=1の場合softmax=1となるため、query/keyによる患者間attentionは働かない。構造と呼出しを変更せず、公開実装に沿う制約として記録する。

## 連続c

観測教師値はCN=0、AD=1。MCIを0.5として追加せず、MMSE/CDRを直接cへ変換しない。論文§3.2の連続pseudo c生成・推定とAlgorithm 2を再現対象に含める。

`paper_cognition.py` は式(4)の前向きc=0/後向きc=1の形状予測平均、端点間の相対位置によるpseudo c、Algorithm 2の区間ごとのc更新を実装する。形状平均の係数は1/2で、cの比による頂点線形補間を代用しない。教師データはtrain患者の端点だけで作り、testの将来端点を使わない。

論文A.3はvoxel入力の単純な3D CNNと連続出力を記すが、層構成・loss・voxel gridは未記載。公開BrainODE codeにも推定器がない。`cognition.py` に以下を補完仕様として実装した。

| 項目 | 実装 |
|---|---|
| CNN | Conv3d 1→16→32→64、kernel3/padding1、ReLU/MaxPool2、global average pooling、Linear→1 |
| cとloss | sigmoid連続出力、BCEWithLogitsLoss、端点0/1とsoft pseudo教師値を保持 |
| voxel化 | 共通faceと固定物理grid。旧surface-fillを比較用に保持し、`center-occupancy` でvoxel中心の内部判定を追加。origin/pitch/grid_shapeと方式を明示し、個別scale・再中心化・cropを行わない |
| 学習・推論 | optimizer/epochsとAlgorithm 2の区間数Nを明示指定。公式ODEへfloat cを入力しRK4で積分 |

数式、soft-label勾配、固定voxel座標、公式ODE接続を検証した。center-occupancyの解析的box・平行移動・crop拒否を含むCNN関連8 testsが通過した。trainだけでfitしたorigin=(-31,-34,-33)仮定mm、128³、pitch0.5仮定mm、3 raysのvoxel候補を全1,405撮像で準備・readback検証した。7 raysとの差は全件0、体積量子化誤差の絶対値は最大0.387%。原著の未公開CNNを完全に復元したとは扱わない。[voxel準備](artifacts/cognition_voxels/adni1_pca150_center05_v1/README.md)

連続c実験 `adni1_continuous_v1` は、trainのCN→AD全4端点組（2患者）からc=0.1〜0.9を生成し、36形状を保存した。test端点を使用しない。自己交差8形状、関係面積最大0.0512%、voxel/符号付きmesh体積差最大0.2964%、3/7 rays差0。実測の1,150 train撮像に全pseudoを加え、CNNは各例1回/epoch・batch16・AdamW5e-4・StepLR10/0.5・固定30 epochs。初期BrainODE100 epochsから、全299実測軌跡と4 pseudo軌跡で追加20 epochs、2患者batch・AdamW5e-4・StepLR20/0.5とする。converter患者内で実測lossと端点組ごとのpseudo軌跡loss平均を1/2ずつ混合し、端点組で異なるcを同一軌跡へ混ぜない。Algorithm 2はN=10。これらの未記載条件は補完仕様で、論文100 epochsとは合計epoch数が異なる。初回CNNのCUDA MaxPool逆伝播の非決定性を同値の非重複max reductionで補い、元設定・停止記録・修正版設定を保持する。[連続c固定設定](artifacts/training_runs/adni1_continuous_v1/README.md)

train22例の診断では旧surface-fillがmesh体積に対して1mmで平均約41%、0.5mmで約20%膨らむことを確認した。voxel中心の内部判定ではPCA meshに対する0.5mm voxel体積差が−0.168〜+0.140%、3/7 raysで判定差0。ray parityが安定しても自己交差曲面の解剖学的妥当性を証明したとは扱わない。[旧方式](artifacts/model_data/adni1_pca150_v1/voxel_diagnostic/REPORT.md)、[中心判定方式](artifacts/model_data/adni1_pca150_v1/voxel_center_diagnostic/REPORT.md)

## 評価と残る作業

主評価は予測meshと元の観測meshの対応頂点間の平均Euclidean距離（仮定mm）。PCA再構成誤差、公開codeのPCA targetへの座標MAE、初期形状を維持する予測の誤差、観測mesh自体のfitting誤差も記録する。`evaluation.py` の4 testsが通過した。観測だけの初期baselineのtest60患者・175将来撮像で、患者macro平均頂点距離は0.3821仮定mm、初期元mesh維持は0.4799仮定mm。予測175形状中102形状に自己交差がある。[初期結果](artifacts/training_runs/adni1_observed_v1/RESULTS.md)

完了範囲は資料・実装調査、台帳とsplit/time作成、ADNI1全1,405撮像のmesh生成と品質flagの詳細確認、trainのみのPCA150 fit、観測だけの初期BrainODE100 epochsと未来予測評価、全voxel準備、train-only pseudo連続c36形状の生成、CNN30 epochs、追加BrainODE20 epochs、Algorithm 2の全175将来target評価・1,750中間入力確認と同じ10段階の固定c比較。実行・集計は2026-10-08 01:00 JSTに完了した。後続cohort比較と非交差surfaceを保証する制約は未実施。

連続c版の最終撮像への患者macro平均頂点距離は0.4332仮定mm（60患者・62撮像）、初期baselineは0.4258で、約1.74%悪化した。同じ10段階の追加ODE・実測c固定は0.4328で、悪化の大半は追加最適化とpseudo導入を合わせた変更で生じ、CNN更新だけに帰属しない。CN/AD識別は撮像micro正解率87.45%、患者macro83.73%、AUC0.9292。pseudo c再現MAE0.22948は一律0.5の0.22222を下回らない。自己交差103/175形状、影響面積最大1.6331%。全中間・最終状態を固定gridで確認したが、ray一致や符号付き体積整合性は解剖学的内部の正しさを保証しない。train converter2患者・test0患者であり、臨床的なconversion一般化の評価ではない。[連続c結果と3D](artifacts/training_runs/adni1_continuous_v1/RESULTS.md)、[cの解釈](artifacts/training_runs/adni1_continuous_v1/cnn/INTERPRETATION.md)

根拠: [BrainODE論文](https://proceedings.neurips.cc/paper_files/paper/2025/file/abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf)、[公開BrainODE](https://github.com/PWonjung/BrainODE)、[公開shape生成](https://github.com/PWonjung/BrainSubcorticalStructure)、[詳細調査](INVESTIGATION.md)。

## 一括生成と再開

`mesh_batch.py` は患者ごとの時系列をGPU workerへ割り当て、同日repeatを日付・UID数値順で全保持する。直前に処理した観測meshを次のsourceにする。同日内順序は論文未記載の補完であり、年齢/timeを変更しない。ゼロ時間差の観測対をODEへ積分しない。

コード・template・公開PointNet・全対象CSVをhome内のsnapshotへ固定する。実行中のコード変更をworkerへ混ぜない。出力manifestを原子的に保存し、再開では設定、環境、コード、元mask、生成artifactのhash、全face/頂点順序、前meshからの引継ぎを確認する。recipe 2の旧QCは比較履歴として保持し、recipe 3のcacheへ混ぜない。

品質flagはDice<0.90、体積差の絶対値>5%、境界平均距離>1仮定mm、表面交差、非閉曲面、maskの分離等を検出する確認用の閾値。cohortの自動除外規則ではない。全候補は元台帳へ対応付け、未完了・失敗も記録する。学習採用は別の確認段階である。

関連18 testsが通過した。実データ5患者・12撮像の一括QCは完了。体積差−1.60%〜+1.34%、Dice0.921〜0.962、表面交差・品質flag0件。全artifact hash、metadata、face/頂点順序、前mesh引継ぎ、copy/in-place再開を確認した。全ADNI生成は完了16撮像を保存して停止し、ADNI1の468患者・1,405撮像の先行生成をGPU4・5の2枚・各4 workerで実行する。各2→4 workerの短い比較でGPU合計速度が約37%増え、完成済み8撮像から同じ計算条件で再開した。[並列数比較](artifacts/mesh_qc/concurrency_2_vs_4/REPORT.md)。一括生成物は `/data5/take/brainode`、軽いQC報告とcode snapshotはhomeに置く。

[12撮像の検証](artifacts/mesh_qc/cohort_pilot_v3/review/REPORT.md)、[ADNI1の最終全件確認](artifacts/mesh_runs/adni1_stage_v3/review/REPORT.md)、[停止した全ADNI処理](artifacts/mesh_runs/adni_all_v3/README.md)。ADNI1の生成は2026-10-07 19:30 JSTに完了し、468患者・1,405撮像、失敗0。元meshの体積差−3.53%〜+2.49%、Dice0.891〜0.973、品質flag73撮像を確認した。初期baselineは全候補を保持する。ADNI2/3・追加cohortの全件品質確認は未完了。
