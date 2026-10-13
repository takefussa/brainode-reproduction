# BrainODE / ADNI1 調査報告

初期調査日: 2026-10-06 (Asia/Tokyo)。論文・公開code・手元データの調査と連続cの実装根拠を記録する。研究元データは読み取りのみ。以後のmesh QC、患者split/time、未完了作業の現状は [SPEC](SPEC.md) と [学習設定](TRAINING_PLAN.md) を参照する。

`/home/take/brainode` の実体とsymlinkでないことを確認してリセットし、公式repositoryと `SPEC.md` を取得・作成した。公式の追跡対象ファイルは未変更。調査用 `.venv` をこのディレクトリ内に作成した。学習環境の完成や学習実行は今回の成果に含まれない。

一次資料:

- [BrainODE 論文 (NeurIPS 2025、付録を含む)](https://papers.nips.cc/paper_files/paper/2025/file/abdc8c031aa6c6917c3b593166e5e340-Paper-Conference.pdf)。同名の2024年の脳信号論文とは別。
- [BrainODE 取得commit](https://github.com/PWonjung/BrainODE/tree/a5e1a2ac2bcbfc1a0aa6cdafafcfdf101135b18d)
- [BrainSubcorticalStructure 取得commit](https://github.com/PWonjung/BrainSubcorticalStructure/tree/d65ffe5da55e53e1431cc93f91e3b348497b8e5d)
- [OpenMAP-T1 取得commit](https://github.com/OishiLab/OpenMAP-T1/tree/628f3646ee69a165e6d85066967fe2b85509ca8a)。これは現在の公開コードであり、2023年のローカルデータ生成に使用されたversionの証明ではない。
- ローカル根拠: `meta/csv/ADNI1/meta.csv`, `meta/json/ADNI1.json`, `meta/level/Level5.txt`, `meta/info/Level5.txt`, `meta/level/Type_Level_num.csv`, `meta/csv/ADNI1/volume_full.csv`。

commit、template hash、metadata hash、集計結果、調査用環境は `audit/` に記録した。

## 1. BrainODE公式実装の実際のpipeline

入力はMRI画像やsegmentation volumeそのものではなく、対応付きmeshの頂点座標列である。

```text
train/test pickle の各subjectの頂点系列 [T, 4470]
  → PCA.transform: [T, 150]
  → 現時点の係数 Λ、time t、cond c を連結
  → ODEFuncWithAttention が dΛ/dt を返す
  → torchdiffeq.odeint(..., method='rk4')
  → PCA.inverse_transform
  → [1490, 3] に戻し、共有face配列で表示
```

公式付属pickleを構造確認のために読んだ結果:

| 項目 | train_hippo.pkl | test_hippo.pkl |
|---|---:|---:|
| 系列数 | 300 | 496 |
| timepoint数2 / 3 / 4 / 5の系列数 | 84 / 56 / 72 / 88 | 124 / 95 / 131 / 146 |
| data | subjectごとのTensor `[T,4470]` | 同左 |
| time | subjectごとのfloat64 Tensor `[T]` | 同左 |
| cond / conv | int32 Tensor `[T]`、確認した全値が0 | 同左 |
| keys | `data`, `time`, `cond`, `conv` | 同左 |

`conv` は `train.py` では使用されない。subject ID、age、撮像日、元meshの由来はこのpickleに含まれない。timeは0〜1で、例は `[0,.25,.5,.75,1]`。これをADNI1の年齢定義の根拠にはできない。

flattenは `[x0,y0,z0,x1,y1,z1,...]` のvertex順を維持する形式。notebookの `reshape(1490,3)` と共有template faceの使い方で確認した。独立meshの任意な頂点順はこの形式を満たしても解剖学的対応を満たさない。

PCAの既定dimensionは150。既定の `--pca=True` では付属PCAを読み込む。`False` 分岐では `traindata['data']` の**各subjectの先頭scanだけ**を積み上げてfitし、全train/test観測をtransformする。testをfitに混ぜる処理はこの分岐にはない。付属PCAは1798 samples / 4470 features / 150 components、whiten=False、保存時sklearn 1.6.1。fit対象subjectやsplitの履歴は取得できなかった。再現対象データでfitする必要があり、付属basisをそのまま評価の根拠にしない。

networkは150係数 + scalar t + scalar c = 152入力。Q/K/Vを各256次元へ射影し、`softmax(Q @ K.T / sqrt(256))`、Linear 256→512、GELU、Linear 512→150で速度を出力する。実際のODE呼び出しではshape batchが1なのでattention行列は1×1で、softmaxは1になる。これはコードの挙動であり、独自architectureへ変更する提案はしていない。

学習は各subjectの全start timepointから、forwardは残りの観測時刻、backwardは過去の観測時刻を逆順に積分する。各trajectoryのcondは**開始scanの値に固定**され、途中の観測診断に切り替わらない。lossはPCA空間のMSEをtrajectoryごとに加算・平均。初期時点だけのゼロloss trajectoryも含む。Gaussian noise (std=.01) とscale `[.98,1.02]` を初期値とtargetの双方へ同じ値で加える処理が既にある。

optimizerはAdamW、lr=.0005、既定1000epochs、StepLRで5分割の節目にlrを半分にする。ただし後述のsample更新条件に問題がある。solverは通常の `odeint` であり `odeint_adjoint` ではない。RK4のstep_sizeは明示されていない。

inferenceの実体は「1観測のPCA係数から、開始condを固定して目的時刻へODE積分しinverse PCAする」。専用inference scriptはない。testでは各start scanから**そのsubjectの最終観測**を予測して誤差を平均する。1-shotと4-shotを区別する専用処理や、複数出発点からの予測を融合する処理はない。

test lossは最終時点のPCA係数MSE。表示名 `MM Loss` の実体は、最終時点の4470個の座標に対する `mean(abs(pred-target))`、つまり座標MAE。targetもinverse PCAされており、元のmeshを保持した `V_real_true` は評価に使われていない。

根拠: [train.py](/home/take/brainode/train.py:50)、[model.py](/home/take/brainode/model.py:62)、[dataset.py](/home/take/brainode/dataset.py:7)、[公式dataset構造の記録](/home/take/brainode/audit/official_dataset_structure.json)、[PCA属性](/home/take/brainode/audit/official_pca_summary.json)。

## 2. BrainODE論文とコードで異なる点

| 論文の記述 / 比較対象 | 公開コード・付属assetで確認したこと | 再現への影響 |
|---|---|---|
| §3.2 pseudo-cognitive sampling / cognition estimator / progressive inference | 該当する生成・CNN・学習・推論がない | 公開ODE部分の再現は論文全体の再現ではない |
| A.4 time = `(age−65)/30`, age 65〜95 | loaderは保存済みtimeをそのまま使用。付属例は0始まりの波番号相当の並びだがage/dateがなく由来は未確定 | 付属timeを研究室データへ流用しない |
| A.4 100epochs | CLI既定1000epochs | paper設定とdemo既定を区別する |
| A.4 shapeのrandom scaling | コードはPCA係数にscaleとGaussian noiseを加える | meanを持つPCAでは係数scaleと実座標scaleは同一ではない。Gaussian noiseの記述も確認できない |
| §4 平均Euclidean距離 | コードは座標MAE | 数値はそのまま論文metricと比較できない |
| ground truthとのshape比較 | 公開testはtargetもinverse PCAする | 実meshとの差とPCA再構成誤差を区別する必要がある |
| 1-shot / 4-shot task、複数観測からの予測融合 | 公開testは各start→finalを個別評価するだけ | 論文taskの専用評価実装は別途必要 |
| PCA150のcompactness約.980 | 付属PCAのexplained variance ratio合計は.95908 | 同じfit populationだという証拠はない。論文の数値をこのassetの属性として扱わない |
| loss式のsquared L2 norm | 実装は次元・時刻についてmean squared error | reductionとtrajectory weightingを記録する |
| NeuralODEのmemoryに関する説明 | 通常のodeintを使用 | adjointを用いた一定memoryの実装とは確認できない |

PCA fit population / subject splitの全履歴、論文全taskを再現する設定は公開コードだけでは確定できない。

静的に確認できる実行・再現上の問題と最小変更候補（未実装）:

| 問題 | 根拠 | 最小変更候補 |
|---|---|---|
| Linuxで付属pickleを開くpathが不適切 | `dataset.py:8-9` のbackslash、checkpoint pathもbackslash | `pathlib` 等によるpath修正だけを行う |
| READMEの `--pca False` がFalseにならない | `argparse type=bool` は非空文字列をTrueへ変換 | 真偽値parserを明示し、研究室PCAをfitする分岐へ正しく入れる |
| pcanumだけ変えると付属PCA150とmodel dimensionが不一致 | `train.py:25,57-59,94` | PCAのcomponent数を検証し、fit / loadを明示 |
| 奇数番目のsubjectを学習更新に使えていない | `train.py:129,184` が共に `i % 2 == 0`。奇数のlossは次の偶数でリセット | 全subjectを更新対象にし、batch2を使うなら正しく蓄積・更新する |
| 環境定義ではない `environment.yml` | 内容はWindows terminalの終了確認文とpromptのみ | 新しい依存定義を用意する。現在のファイルでconda環境を作れるとは扱わない |
| 可視化notebookが存在しないasset名を参照 | `test_hippo_30.pkl` はrepositoryにない | 実際のdataset pathとLinux pathへ最小修正 |
| Python randomだけseedなし | scaleに `random.uniform` を使うがrandom.seedなし | 再現性のためseedを統一する |

これらはcode inspectionの結果であり、学習を実行して取得したerror logではない。今回は公式ファイルへ修正を適用していない。

## 3. BrainSubcorticalStructureのmesh生成pipeline

**BrainODEの `hippocampus_template_mesh.obj` と、このrepositoryの `whole_brain_structure/temp_meshes/17_temp.obj` は同一ファイル。** 両者とも1490 vertices / 2976 triangles。頂点・face配列、ファイルSHA256まで一致する。SHA256は `dc74b3aee2420c7c600812b54aa0891eef64f88eb2137f131a450da7d436a8a1`。templateの頂点重心はほぼ原点。

template生成notebookが示す方法は、複数のSynthSeg maskからregionを抽出し、marching cubes / smoothingでsurfaceを作り、point-to-plane ICPで整列、occupancyを平均しthreshold .3、closing、surface抽出・Taubin等のsmoothing・目標edge 1.5でremeshし、重心を引いてOBJ保存する流れ。外部 `areas.pkl` 等に依存し、左右のclass指定やmirror保存にも不整合があるため、提供OBJの全生成履歴がこのnotebookだけで再実行可能とは確認できなかった。今回提供templateを再生成する必要はない。

individual fittingの入力生成は `optim_data_generator.ipynb`:

1. segmentationのregion labelをbinary maskへ変換する。
2. `util.voxel_to_bd_pcd` が2×2×2 convolutionの境界（0より大きく8未満）からpoint cloudを作り、座標に.5を加える。
3. target point cloudの平均座標を引いて中心へ移す。
4. point-to-point ICPでtargetをtemplate、または前時点のmeshにrigid alignmentする。
5. pickleへ `vertices: [N,3]`, `faces: [F,3]`, `target: [M,3]` を保存する。

targetは不定個数の境界点。BrainODEへ渡すのはtarget点群や独立marching-cubes meshではなく、**固定topologyの変形済みtemplate vertices**。このutilityはvoxel indexから点を作り、NIfTI affineを適用していないため、1mm voxelという前提とaxis方向の扱いを研究室データで確認する必要がある。

`optimize_p2p.py` は個別scanに対してPointNetOptを初期化し、vertex displacementを出力するmodelをAdamWで最適化する。距離lossはpoint-to-face×3 + Chamfer×.5。regularizationはedge項×1500 + uniform Laplacian×5 + normal consistency×1、さらに元verticesからのMSE displacement×1。edge項の実装はweighted squared edge lengthの標準偏差で、名称から単純なedge-length varianceだと決めつけない。出力は原則 `[1,N,3]` の`.npy`、500epochごとおよび最終epochに保存。faceは入力の共有topologyを別途使用する。

longitudinalでは初回をtemplateからfitし、次回は前回の変形verticesを初期値にする。論文A.1とnotebookのscan2以降に明示される。scan3で前時点がなければさらに以前へfallbackする例もあるが、同じ方針を研究室データに最終決定したわけではない。

固定face・頂点数・indexは保持される。一方、lossは解剖学的landmarkを強制せず、厳密な解剖学的対応、vertexの滑り防止、self-intersection回避を保証するものではない。longitudinal初期化とregularizationは対応の安定化に寄与する設計であり、QAが必要。

公開meshコードの未解決点:

- `models/loss.py:17` が存在しない `models.point_mesh_loss2` をimportする。未使用importでも実行を妨げる。
- `run_optimization.py` は既定label='17'で `label_data` が未定義になり、外部の未収録 `../../voxel2mesh/hippo_p2p.py` を呼ぶ。提供コードの実行wrapperとしてそのまま使えない。
- `optimize_p2p.py` はhard-coded CUDAと外部directory階層に依存する。
- `pred_mesh` はloop内で更新されず、毎epochのsource verticesは初期meshのまま。paper Fig.4のepoch内累積vertex updateとは同じ実装でない。
- paper A.1にあるnormal displacement項は提供lossに見当たらない。論文を再現基準とし、公開lossだけを理由にこの項を省略しない。論文で実装に必要な定義が足りなければ、補う前に相談する。

根拠: [template比較](/home/take/brainode/audit/template_comparison.json)、[notebookコード抽出](/home/take/brainode/audit/mesh_notebook_source.txt:771)、[util.py](/home/take/brainode/references/BrainSubcorticalStructure/whole_brain_structure/util.py:36)、[optimization](/home/take/brainode/references/BrainSubcorticalStructure/optimize_p2p.py:99)、[mesh loss](/home/take/brainode/references/BrainSubcorticalStructure/models/loss.py:21)。notebookは実行せずcodeを読んだ。

## 4. ADNI1/fullの実際の構造

```text
/data2/radiology_datas/clean3/ADNI1/full/
  rigid/<uid>.nii  (2402 files)
  trsaa/<uid>.nii  (2402 files)
```

ファイル名は数値UID。ファイル名だけからsubjectやvisitは得られない。`meta.csv` のuidをjoin keyとして `pid`, `day`, `age`, `class` に対応付ける。研究室metaの `path_full` はrigid、`path_full_mni` はtrsaaを指す。CSVとJSONは2402レコードで完全一致する。

TRSAA / rigidはそれぞれ全2402scanで同一geometry、float32、shape `(160,224,160)`、spacingの数値 `(1,1,1)`、RAS。affine:

```text
[[1,0,0, -80],
 [0,1,0,-112],
 [0,0,1, -80],
 [0,0,0,   1]]
```

qform code=0 / sform code=2 (aligned)。全件headerの分布は `audit/adni1_geometry.json` に保存済み。**全件でspatial unitsはunknown。spacingの数値は1.0と確定できるが、headerにmm単位が明記されているわけではない。1mmという解釈は処理履歴からも確認する必要がある。**

## 5. TRSAAとして使うべきファイル

候補は **`ADNI1/full/trsaa/<uid>.nii`**。metadataの `path_full_mni` と全件一致する。half、rigid、unifiedをTRSAAとして代用しない。

「MNI」の根拠はmetadataのfield名。headerから確認できるのはspacing数値1.0、RASの固定gridであり、sform codeはMNIを直接指定する4ではなく2。ローカル `MNI/MNI_FULL.nii` は同じshape・spacingだがtranslationは `(-80,-129,-72)` で、TRSAAのaffineと一致しない。

TRSAAの正確な変換（rigid / affine scaling / nonlinear）、生成script、変換matrix、元scanとのmappingは取得できなかった。metadataの `preprocessing` はADNI由来のGradWarp / B1 / N3 / Scaled等であり、研究室TRSAA生成履歴の証明には使えない。形状を変える変換があるならbaseline比較へ影響する可能性はあるが、名前からそうだと断定しない。

BrainODEは直接MRIを入力しないので、TRSAAとparcellationのgrid差だけを理由に追加registrationする必要があるとは結論していない。

## 6. ADNI1/parcellationの実際の構造

`ADNI1/parcellation/<uid>.nii` のflat directory、2402 files。全件uint16、shape `(256,256,256)`、spacingの数値 `(1,1,1)`、RAS、qform=0 / sform=2、spatial units=unknown。affineを含むgeometry signatureは2401種類。

代表affine（微小なrotation項は読み取った値を丸めて表示）:

```text
[[1,0,0,-129.891098],
 [0,1,0, -94.430923],
 [0,0,1,-127.472839],
 [0,0,0,   1]]
```

全件headerを集計し、voxel dataは数値UID順から均等に選んだ12scanで確認した。各scanのunique labelは277または278種類。12scanのunionは0〜280から235 / 269 / 271を除いた278種類。**12scanのlabel調査を全2402scanのvoxel検証と表現しない。** 左海馬75は全12scanに存在し、2572〜3847voxel、RAS world座標の重心xは全例で負だった。左海馬に加え、280領域すべてのvoxel数が各scanの `volume_full.csv` と一致した（不一致0 / 3360照合）。これはlabel体系とCSVの対応を裏付けるが、unit fieldや生成softwareの証明ではない。

実label値・欠落label・海馬voxel数・centroidは [label sample記録](/home/take/brainode/audit/adni1_label_samples.json)、全領域照合は [volume照合](/home/take/brainode/audit/adni1_label_volume_validation.json) に保存した。計算上のmm³値はspacingがmmという仮定付きのfield名で記録している。

## 7. fullとparcellationをどのように対応付けられるか

`trsaa`, `rigid`, `parcellation`, `meta.csv` のUID集合は全件一致し、重複UIDは0。**UIDで2402組を一対一に対応付けられる**。subject groupingは `pid`、順序は `day` を使う。元のvisit/session codeはこのmetadataにないため、撮像日順のordinalを元visit codeだと扱わない。

ファイルの一対一対応と画像の空間対応は別。**全2402組でMRIとmaskはshapeもaffineも異なり**、同じvoxel indexで重ねられない。NIfTI affineによる座標mappingだけで十分なのか、TRSAA生成時の別transformが必要なのかは未確定。追加registration / resampling / segmentationやり直しをまだ実施しない。

meshはparcellation自身のgeometryを基準に作ることが候補。TRSAAは対応scanのMRI確認や可視化に使う。これにより、ODE入力のためだけにMRI全体を加工する処理を増やす必要はない。

## 8. parcellationがOpenMAP-T1由来か

**label体系はOpenMAP-T1のType1 Level5と一致。実際の生成software / versionは未確定。**

根拠はローカルLevel5辞書の全280行が公式 `level/Type1Level5.txt` と一致すること、multilevelのtype/level mapping、volume CSVのROI名、およびclean3内にOpenMAP-T1/V2のCNet/PNet/SSNet/HNet weightsがあること。モデルweightの存在だけでは、このADNI1 outputをそれで作った証明にならない。

clean3と関連する既存projectの必要な範囲でscriptを検索したが、ADNI1の生成run log、model version、source-output transform履歴は見つからなかった。現在のOpenMAP-T1 READMEの出力space仕様を古いローカルデータへ当てはめない。

## 9. left hippocampusのlabel

今回のLevel5 parcellationの左海馬は **75 = Hippo_L**。右海馬は76。この対応はローカルのLevel5定義、Type_Level_num/name mapping、OpenMAP-T1公式辞書で確認した。

**17はこの辞書ではLFOG_Lであり左海馬ではない。** BrainODE / SynthSegでのtemplate label17と、研究室maskの抽出label75を混同しない。

Fimbria_L=269 / Fimbria_R=271は辞書上のLevel5では別領域だが、確認した12scanでは両labelが出現しなかった。Type1 Level4等では海馬へ統合するmappingがある。Level5で75へ269を足す処理は未決定であり、追加していない。SynthSeg海馬とのregion definitionの差もbaselineの限界として記録する。

## 10. longitudinal subject数とtimepoint分布

2402scan / 687subject。同日scan重複は83組、subject×別撮像日の総数は2319。ファイル数をそのままtimepoint数にしない。

| 条件 | ファイル数で数えたsubject | 別撮像日で数えたsubject |
|---|---:|---:|
| 2以上 | 565 | 565 |
| 3以上 | 436 | 436 |
| 4以上 | 296 | 292 |
| 5以上 | 184 | 176 |

| timepoint数 | ファイル数の分布 | 別撮像日の分布 |
|---:|---:|---:|
| 1 | 122 | 122 |
| 2 | 129 | 129 |
| 3 | 140 | 144 |
| 4 | 112 | 116 |
| 5 | 80 | 87 |
| 6 | 46 | 48 |
| 7 | 21 | 21 |
| 8 | 17 | 10 |
| 9 | 12 | 8 |
| 10 | 3 | 1 |
| 11 | 4 | 1 |
| 12 | 0 | 0 |
| 13 | 1 | 0 |

別撮像日の隣接間隔は1632組、min=1日、25%=177日、median=202日、75%=362日、max=1506日、mean=253.72日。1〜30日の短い間隔も108組あり、別日であっても同visitのrepeatかどうかは未確定。単純な6か月等間隔として扱わない。同日scanをそのまま使うとtimeが重なり、ODEのstrictly monotonic time要件に合わない。

根拠: [metadata集計](/home/take/brainode/audit/adni1_metadata_summary.json)、[subject一覧](/home/take/brainode/audit/adni1_subjects.csv)、[全scan対応表](/home/take/brainode/audit/adni1_scans.csv)、[time QA](/home/take/brainode/audit/adni1_time_qc.json)。まだ重複を除去してdatasetを作成したわけではない。

## 11. age / scan date / diagnosisが取得可能か

`meta.csv` から取得できる。ageとdayは2402scanすべてにあり、ageは54.6329〜93.0219歳、撮像日は2005-08-26〜2010-09-02。`class` はCN=712scan、AD=693、MCI=977、欠損20。`class_base` はCN / AD / LMCIを含み、17欠損。撮像時の診断とbaseline診断を取り違えない。

MMSEは25scanで欠損、MoCAは全scan欠損。元visit codeはない。年齢差と日付差は概ね整合するが、同日に年齢が異なる1組があり、撮像日順に並べたconsecutive ageでは82組の同値と1組の逆転がある。年齢が全件あるだけで、そのままODEのtimeとして有効とは結論しない。

## 12. BrainODEのtimeの論文仕様と未確定の処理詳細

論文A.4は **65〜95歳のsubjectを選び、年齢を `t=(age−65)/30` で正規化する**。研究室データへの適用設定では年齢上下限で除外せず、全CN/ADを対象に `t=(age−train_min_age)/(train_max_age−train_min_age)` を使う。trainから求めた1つの共通係数をtest/推論にも固定する。患者ごとのbaseline=0やvisit番号を代用しない。[最新のtime定義](audit/time_definition.md)

ADNI1には65歳未満のscanが146ある（MCI等も含む）。以前の撮像単位65〜95歳・CN/ADの台帳から、年齢上下限を外した結果、候補は61撮像増えて468人・1405撮像となった。metadataの撮像時年齢を使い、clipしない。現行設定は同日repeat全保持、RID単位80%/20%split、PCAはtrain全採用撮像である。初回年齢 + 実撮像日差/365.25による年齢再構成は採用しない。train限定の正規化は実装・検証済みで、ADNI全phaseの患者split後にtrainだけのtime係数をfit済み。[最新台帳](artifacts/adni1_allage_eligibility/summary.json)

## 13. BrainODEのcondの論文仕様と未確定の選択規則

論文の観測診断の端点への対応は **撮像時classのCN→0、AD→1**。ただしcondはこのbinary対応だけで完成しない。論文§3.2の連続cの生成と推定を含める。論文§5はMCIをmodelingから除外すると明記している。MCI除外を未決定扱いしていた先の説明は訂正する。MCIを.5とする規則は採用しない。MMSEの0〜1変換をcの代用にしない。

§3.2の式(4)では、同一患者のNC時点 `t0` とAD時点 `t1` の間の時刻 `s` で、`c_tilde=(s-t0)/(t1-t0)` を教師値とする。pseudo形状はNC端点からの前向きODE予測（c=0）とAD端点からの後ろ向きODE予測（c=1）の **1/2ずつの平均**。観測頂点をcの比で線形補間する方法ではない。

実観測とpseudo形状・教師値でcognition estimatorを学習する。付録A.3は **voxel形状→3D CNN→連続c** と明記している。本文の `tau(Lambda)` の記法だけを根拠にPCA係数入力のMLPに置き換えない。Algorithm 2は等間隔に分けた各区間の開始時に現在の予測形状からcを再推定し、そのcで次区間へODE積分する。

CNNの層構成、損失、学習設定、PCA逆変換後のvoxel化条件、pseudo生成密度、観測/pseudo混合比、初回/追加学習の配分、一般のNは記述が足りない。公式mainは取得済みcommitと同じで、これらの実装はない。A.3の引用研究Liu et al. (2020)はabstractを確認した範囲ではMRIと複数CNNを用いる別手法であり、そこからBrainODEの具体的な簡易CNNの仕様が一意に決まるとは言えない。本文の取得はできていない。

数式・手順を [paper_cognition.py](/home/take/brainode/paper_cognition.py)、voxel入力3D CNNと連続出力、soft-label lossと学習関数、固定gridのvoxel変換、公式ODEへのRK4接続を [cognition.py](/home/take/brainode/cognition.py) に実装した。CNNチャンネル16/32/64、ReLU/MaxPool、sigmoid、BCE、surface voxelize+fillは補完仕様として [SPEC](/home/take/brainode/SPEC.md) に明記した。voxel grid、N、推定器epochs/optimizer、cohort/split/混合比は必須指定または未確定。原著の未公開CNNを忠実に復元できたという意味ではない。

解析解のあるODE、MCIの非採用、fractional教師値の勾配、voxelの固定座標と内部fill、公式modelでの逆伝播を [検証](/home/take/brainode/tests/test_cognition.py) し、14件すべて通過した。[検証ログ](/home/take/brainode/audit/cognition_tests.txt) を保存した。単体検証に使った箱形状は合成fixtureで、研究データのmesh生成や再現スコアではない。BrainODE付属demoでは学習も評価も行っていない。実modelでの研究データのc生成・学習・予測はまだ実行していない。

連続cに必要なconversion例の確認として、論文B.1が挙げるcohortのmetadataまで調査した。以下は文字列がそのまま `CN` と `AD` の両方あるsubjectの集計であり、採用条件でも全conversion例の確定数でもない。診断文字列の再分類は行っていない。

| cohort | metadata scan行数 | subject数 | 同じsubjectに文字列CNとADの両方あり |
|---|---:|---:|---:|
| ADNI1 | 2402 | 687 | 2 |
| AIBL | 1104 | 551 | 2 |
| OASIS2 | 1368 | 150 | 0 |
| OASIS3 | 789 | 614 | 3 |

ADNI1は撮像日でNC→ADを確認できる2人・端点4組で、両端年齢はすべて65〜95。そのうち1人は途中にMCIがある。OASIS2の0はmetadataのclass文字列に基づく値であり、元の診断履歴でconversionがないとは判断しない。OASIS3には年齢欠損179行と複数の診断表記があり、独自にCN/ADへまとめない。他cohortの画像・meshは未調査。根拠: [metadata調査](/home/take/brainode/audit/continuous_c_metadata_inventory.json)。

調査時に「全期間CNまたは全期間AD、別撮像日2以上」という仮条件で275人、さらに「全観測年齢65〜95」を加えて262人と集計した。これらの追加条件は論文の明記された選択アルゴリズムではなく、採用条件にしない。262人という仮集計を学習対象の決定に使わない。Table 5のADNIのCONV=0は構成の報告であり、それだけでは診断変化subjectを除く規則を特定できない。全2402scanの調査台帳を保持し、対象dataset・train/test splitはまだ作成していない。

MCIを含むsubject全体を除くかMCIのscanだけを除くか、CN→AD等の診断変化、診断欠損の処理は論文の記述だけでは確定できないため相談する。固定condの公開ODEと§3.2のcognition estimator付き方法を区別し、論文にあるcomponentを公開コードにないという理由だけで省略しない。

testの未来診断をinputとして与えない。評価時のsubject leakageとPCAへのtest混入を避ける必要があるが、実際のsplitとPCA fit populationは論文で確認できない詳細を相談して確定する。

## 14. 仕様で区別する論点

SPECでは論文を再現基準とする。MCI除外と年齢正規化の明記された仕様、未記載の選択規則を区別し、262人という診断一定の仮集計を採用条件にしない。以下の実装上の詳細は、論文の記述を確認し、不足があれば相談する事項である。

- 「baseline」の範囲: 論文の§3.2は既存componentであり新規改良ではない。公開コードにないcomponentは必要な仕様を確認し、不足を相談する。公開ODEだけで論文全体を再現したとは扱わない。
- 「対応付きmesh」: face / index固定は必要条件、厳密な解剖学的対応はQA対象であることと、前時点meshによるlongitudinal初期化を明記する。
- 「不要な処理を入れない」: 公式fittingの中心合わせ・rigid ICP・regularizationと、公式trainingに既にあるaugmentationを、追加の前処理と区別する。どのpaper/code設定を残すかは差分として説明する。
- 「MRI→parcellation」: UID対応だけで同じ画像gridだと考えず、mask geometryとTRSAAの生成transformを確認してから座標を扱うことを明記する。
- 「再現評価」: coordinate MAE / PCA-reconstructed targetと、paperのEuclidean distance / 元meshのground truthを区別して報告する。metricの追加は新しいtraining lossの追加ではない。
- 「time / cond / cohort」: 年齢範囲とMCI除外は論文仕様として確定済み。適用単位、同日repeat、診断変化・欠損、metadata ageの不整合等の未記載の規則を相談して明記する。

Linux path、CLI、環境定義、学習sample更新、meshコードの欠落importの修正は実行に必要な候補。loaderの選択規則、split / PCA fit、normal displacement lossやepoch内vertex更新は論文の記述を確認し、不足する詳細を相談してから扱う。

## 15. 初期の座標QCで確認する項目

**1subject・異なる2撮像日の左海馬maskと座標を確認し、最小のmesh fitting入力仕様を確定する。** まだ大量mesh、PCA fit、BrainODE trainingを行う段階ではない。

この1段階で、同日repeatのないscan pair、UIDとmetadataの対応、label75の位置、maskのmm / axis、TRSAAとのspace関係、初回templateと次回前時点meshという初期化を確認し、必要なcode修正だけを具体化する。既存transformの有無を確かめる前にMRI registrationや再segmentationを追加しない。

初期QCの対象は1subject・2撮像日の左海馬maskと座標。TRSAAとparcellationのgrid差、longitudinal初期化、公開mesh codeの不足を整理する検証単位とした。

この初期QCは完了済み。現在の状態と設定は [SPEC](SPEC.md)・[学習設定](TRAINING_PLAN.md) を参照する。

## 追加調査: ADNI全体・AIBL・OASIS2/3

2026-10-06、手元の6cohortの全15,727 metadata行を読み取り専用で確認した。全行を保持し、文字列CN/AD・有限年齢・画像ファイル・撮像日を照合できるのは9,312撮像だった。元データは変更していない。診断表記の再分類・年齢補完・train/test作成・実学習は行っていない。[全件調査報告](artifacts/multicohort_inventory/REPORT.md)

ADNIはphaseを跨ぐRIDをまとめるとCN/AD1,455人・6,746撮像で、CN→ADの端点を持つ24人がある。ADNI1の2人に比べ、連続cの教師データを増やすうえで有力。AIBLはMRI日の進行とage差の不整合、OASIS2はCN/CDRの対応とrepeat、OASIS3は欠損年齢と複数診断表記の確認が必要。OASIS公式資料は2/3の患者重複の可能性から個別使用を記載しており、各datasetを追加する別実験を推奨する。

追加cohortで各3例、計18例のmask内容も調べ、label75の存在と共通header条件を確認した。全parcellationの品質やTRSAAへのregistrationを確認したという意味ではない。meshの初回体積差は撮像ごとの当てはめ段階の問題で、BrainODE/CNNの学習データ増加とは分けて対応する。
