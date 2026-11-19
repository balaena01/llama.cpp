> このリポジトリでの対応: 文中の `scripts/` は `ps4/scripts/`、`demo/` は `ps4/demo/`、`experiments/` は `ps4/experiments/`。`patches/sensenova-loop-ps4.patch` はこのブランチのコミットそのもの。

# PS4でSenseNova-Loop-9Bを動かす（PS4 Linux + Vulkan + llama.cpp）

PS4の初代・Slim系GPU（GCN2世代のLiverpool、18CU、GDDR5共有8GB）で、ループ構造とmHCを持つ新しい9Bモデル **SenseNova-Loop-9B** を動かしたときの作業記録。
Garlic-LLM（PS4用の自作LLMアプリ）の考え方を参考にしつつ、**PS4のGPUのメモリを最大限使う**ことを目標にした。

- 参考: https://huggingface.co/garlic-llm/garlic-llm-ps4
- モデル: https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF
- 推論エンジン: OpenSenseNova/llama.cpp（commit `24ea7bb`）に、PS4向けの独自パッチを追加

最終ゴール: PS4を **llama-serverのAPIサーバー** にして、PCから使う。速度が分かる動画と、このワークフローの解説を載せたX用の動画を作る。

---

## 0. 結果サマリ（2026-11-19時点）

| 段階 | 1トークンの生成時間 | 速度 |
|---|---|---|
| 初めて動いた（一部の重みはCPU、GTT 3.4GB） | 720 ms | 1.4 tok/s |
| GTTを5.5GBにして全部GPUに載せた | 612 ms | 1.6 tok/s |
| 重みをキャッシュなしGTT（Garlic経路）に配置 | 200 ms | 5.0 tok/s |
| mHCの経路計算もGPUで実行（CPUとの行き来が97回から2回に） | 145 ms | 6.9 tok/s |
| PS4向けにIQ4_NLで量子化し直し（imatrix、日英中の調整データ） | 118.7 ms | 8.4 tok/s |
| Flash Attentionを切る（`-fa off`） | 116.1 ms | 8.6 tok/s |
| routerを1つのカーネルに統合（RMS正規化＋行列計算＋バイアス＋Sinkhorn） | 111〜113 ms | 8.9〜9.0 tok/s |
| Sinkhornのsoftplusを半分に＋出力ヘッドもIQ4_NL（ps4c） | **108.3〜108.6 ms** | **9.2 tok/s** |
| （比較）元の実装（mHCの融合なし）、Q4_K_M | 507 ms | 2.0 tok/s |

- 出力は融合版と元の実装で一致することを確認済み。
- 計算グラフの演算の数: **52,749個から4,161個**に減った（mHCのSinkhorn処理を1つにまとめた効果）。
- 最終版のモデル（IQ4_NL-ps4c）の品質は、公式のQ4_K_Mと同じ（perplexity 14.84 対 14.87、どちらも±0.65）。サイズは4,959 MiBで、11%小さい。
- MTPは、IQ4_NL版で測り直しても逆効果（Step 12）。行列×ベクトル計算の専用シェーダーは4種類試したが、汎用に勝てなかった（Step 14）。

---

## 1. 環境

### PS4側
| 項目 | 内容 |
|---|---|
| 本体 | PS4（GPUのPCI ID 1002:9923、southbridge Baikal） |
| GPU | AMD Liverpool（GCN2 / GFX7）、FP16演算なし、int8内積命令なし、subgroup 64 |
| OS | CachyOS（Dionさん版のLXQt移植）、内蔵HDDに `/user/app/linux.img` として配置 |
| カーネル | 7.3.3-Strawberry（ArabPixelさんたちのPS4 Linuxカーネル） |
| Vulkan | Mesa 26.0.4 RADV（`mesa-ps4`） |
| ローダー | GoldHENからペイロードを起動。起動ファイルは `/data/linux/boot/`（bzImage, initramfs.cpio.gz, bootargs.txt） |
| メモリ | GDDR5 8GB = VRAM 1GB（ペイロードの初期値）＋Linuxから見えるRAM 約6.9GB |

### PC側
- Windows 11、SSH接続（`ssh -i ~/.ssh/ps4_ed25519 user@192.168.0.5`）
- PS4のログインシェルはfishなので、スクリプトは `bash -s` で流す

### モデル
| 項目 | 値 |
|---|---|
| 構造 | 32層のうち4〜27層を同じ重みで2周する（実行は56層分）＋2ストリームのmHC（Sinkhorn 20回） |
| 注意機構 | 4層ごとに、線形注意（Gated DeltaNet）3層＋通常の注意1層 |
| 語彙 | 248,320 |
| Q4_K_M | 5.41 GiB（GPUに載せる分 4.75 GiB、token_embdの0.53 GiBはCPU側） |
| 1トークンで読む重み | 約7.7 GiB（2周する分も含む） |

---

## 2. 作業の流れ

### Step 1: 下調べ
- Garlic-LLMのページから分かったこと: Qwen3.5-9Bを「ほぼ4bit」の独自形式に変換し、PS4で31〜39 tok/s出している。手法は非公開。
  - 名前の由来はおそらくPS4のメモリバス。**Garlic**はGPU直結の速い経路（176GB/s）、**Onion**はCPUキャッシュと同期する遅い経路。
- SenseNova用フォークの差分は、モデルの処理の組み立て（`src/models/sensenova-loop.cpp`）だけ。GPU側の新しい演算はない。
  - Qwen3.5系の演算（Gated DeltaNet、ssm_conv、softplusなど）はVulkan版にすべて揃っている。
- 問題点: mHCのSinkhorn正規化（2×2行列を20回反復）が細かい演算の積み重ねで書かれていて、1トークンあたり約49,000個の小さなGPU処理になる。

### Step 2: SSHの準備
- パスフレーズなしのPS4専用鍵 `ps4_ed25519` を作って登録した。

### Step 3: PS4上でビルド
- `scripts/build_sensenova.sh`（Vulkan有効、GGML_NATIVE）
- つまずき: llama-serverのWeb画面ファイルのダウンロードで失敗した。`-DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF` で回避。
- ビルド時間: Jaguar 8コアで約20分。差分ビルドなら約5分。

### Step 4: mHCの融合（今回一番の改造）
- 2本のストリームを1つのテンソル `[4096, 2, T]` にまとめた。
- 新しい演算 **`GGML_OP_MHC_ROUTE`** を追加した（CPU版とVulkanシェーダー版）。
  - 入力は経路のロジット8個。出力は、分配の重み（sigmoid）、合流の重み（2×sigmoid）、Sinkhornの結果（exp）。
  - Sinkhornは元のグラフと同じ順序で計算する（対数のまま正規化、softplusの式も同じ）。
- 合流と分配には、既存のDeepSeek-V4用の `dsv4_hc_post` / `dsv4_hc_pre` を使った。ストリーム数が4に固定されていたのを、2でも動くように汎用化した。
- `LLAMA_SENSENOVA_MHC_FUSED=0` で元の実装に戻せるようにした（比較用）。
- `test-backend-ops` で、VulkanとCPUの結果が一致することを確認した。

### Step 5: GPUメモリの壁（GTTの拡張）
- 初期状態でVulkanから使えるのは、VRAM 1GB＋GTT 3.37GB（`ttm.pages_limit` の初期値はRAMの50%）で、合計約4.4GB。モデルが載らない。
- amdgpuドライバはカーネルに組み込まれているので、起動オプションでしか変えられない。
- PS4を通常モードで起動し、GoldHENのFTP（ポート2121）で `/data/linux/boot/bootargs.txt` を置いた（`scripts/bootargs.txt`）。
  ```
  （既存の起動オプション） ttm.pages_limit=1441792
  ```
- 結果: GTTが5.5GBになり、Vulkanから使えるメモリの合計は**6.5GB**になった。
- ついでにデスクトップを止めた（`sudo systemctl set-default multi-user.target`）。約0.8GB空き、PS4にキーボードを刺さなくてもSSHだけで使える。

### Step 6: GPUが止まる問題（行列×ベクトル計算）
- 症状: 推論を始めた直後に `ring comp_1.0.0 timeout` が出て、GPUがリセットされる。
- 切り分け: 既存の融合処理、自作のmHC融合、Flash Attentionのどれを切っても再現した。そこで演算を個別に試した。
- 原因: **2〜8トークンをまとめて計算する行列×ベクトル計算**で、1ワークグループの「行数×列数」が16を超えると固まる。
  - K系の量子化（4行担当）は5列以上、IQ4_XS（8行担当）は3列以上で固まる。
  - プロンプトが約6トークンだったので、ちょうどこの範囲に入っていた。メモリの量とは無関係。
- 対策: PS4のGPUを検出したら、まとめて計算する列数の上限を2にする（`GGML_VK_MUL_MAT_VEC_MAX_COLS` で変更可）。

### Step 7: メモリの置き場所（GarlicとOnion）
- 全部GPUに載せても612ms/トークンと遅かった。処理ごとの時間を測ると、層ごとの行列計算が単体テストの約4分の1の速度しか出ていなかった。
- 原因: llama.cppは初期設定で「GPU専用（device-local）」のメモリを優先する。VRAMの1GBからあふれた重みが、**遅い経路（Onion、CPUキャッシュ同期）** のメモリに置かれていたと見ている。
- 対策: キャッシュなし（ライトコンバイン）のGTTに明示的に置く（`GGML_VK_PREFER_HOST_MEMORY`）。PS4では自動で有効にした。
- 結果: 行列計算が**3.9〜6倍**速くなり、612msから200msになった。

  | 行列×ベクトル計算（1トークン） | 初期設定 | キャッシュなしGTT |
  |---|---|---|
  | FFNのgate（Q4_K 12288×4096） | 1775 µs | 458 µs |
  | FFNのdown（Q6_K 4096×12288） | 2728 µs | 457 µs |
  | 4096×4096（Q4_K） | 603 µs | 189 µs |

### Step 8: mHCの経路計算がCPUに回されていた
- 経路の行列計算（F32、8×8192）には精度の指定（`ggml_prec_set_src(F32)`）が付いていて、Vulkan版は一律で「非対応」にしていた。そのせいで1トークンあたり97回、CPUとGPUの間で同期と転送が起きていた。
- 対策: FP16を持たないGPUでF32×F32の場合は、F32のまま計算されるので対応扱いにした。
- 結果: 行き来が97回から2回になり、200msから**145ms**になった。

### Step 9: VRAMも使う
- キャッシュなしGTTだけに置くと、VRAMの1GBが空いたままになり、システムRAMが足りなくなった（imatrixの計算中にOOMで落ちた）。
- 対策: VRAMに収まる分（VRAMの容量から192MBを引いた量）を先にVRAMに置き、残りをGTTに置く（`GGML_VK_VRAM_FIRST_MB` で変更可）。
- 結果: VRAMに820MiB載り、速度は変わらない（148ms/トークン）。

### Step 10: 量子化形式の比較（PS4のGPUでの実測）
4096×14336の行列、1トークン:

| 形式 | 時間 | bit数 |
|---|---|---|
| **IQ4_NL**（4行担当） | **402 µs** | 4.5 |
| Q4_0 | 434 µs | 4.5 |
| Q4_1 | 500 µs | 5.0 |
| Q6_K | 515 µs | 6.6 |
| Q4_K | 523 µs | 4.5 |
| Q8_0 | 517〜731 µs | 8.5 |
| Q5_K | 702 µs | 5.5 |
| F16 | 813 µs | 16 |

- 分かったこと: PS4のGPUは **bit数ではなく、展開の計算の重さで速度が決まる**。f32なら146GB/s出るので、メモリ帯域が限界なわけではない。
- IQ4_NLは表を引くだけで展開できるので軽く、Q4_Kより約23%速い。2トークン同時でも1.4倍の時間で済む（MTPに向いている）。
- 方針: 「ほぼIQ4_NL」に量子化し直す。出力ヘッドは品質のためQ6_K、attn_vはQ5_K（llama-quantizeの規則）、token_embdはCPU側なのでQ4_K。

### Step 11: imatrixと量子化の作り直し
- PCのドライブの空きが少ないので、量子化はPS4上でやった（PS4は空き269GB）。
- BF16モデル（18.4GB）をPS4にダウンロードした。途中で2回切れたので、`--http1.1` と再開のループで取り直した（`dl_bf16.sh`）。sha256は公式と一致。
- imatrixの調整データ = bartowskiさんの `calibration_datav3`（英語）＋Wikipediaの日本語と中国語（`scripts/fetch_wiki.py`）。言語ごとに交互に並べた（`scripts/calib_small.txt`）。
- imatrixは今あるQ4_K_MでPS4のGPUを使って計算した（`scripts/run_imatrix.sh`）。512トークンあたり22.7秒、100チャンクで約38分。
- つまずき1: VRAMを使わずGTTだけに重みを置いていたので、システムRAMが足りずOOMで落ちた。Step 9のVRAM優先で解決。
- つまずき2: 自動の配分だと `attn_qkv_gate`（792MiB）がQ5_Kになる。Q5_KはPS4で遅く（702µs）、4列で止まるので、IQ4_NLを指定し直した。
- つまずき3: 量子化とperplexityの測定を同時に動かしてOOMになった。llama-quantizeは出力ヘッドの変換で約4GB使うので、単独で動かすこと。
- 最終的なコマンド（`scripts/quantize_iq4nl_b.sh`、約15分）:
  ```sh
  llama-quantize --imatrix imatrix-sn-loop.gguf \
    --output-tensor-type q6_K --token-embedding-type q4_K \
    --tensor-type attn_v=q6_K --tensor-type attn_qkv_gate=iq4_nl \
    SenseNova-Loop-9B-BF16.gguf SenseNova-Loop-9B-IQ4_NL-ps4b.gguf IQ4_NL 8
  ```
- 結果:

  | モデル | perplexity（wikitext-2、24チャンク） | サイズ | 速度 |
  |---|---|---|---|
  | Q4_K_M（公式） | 14.87 ±0.65 | 5,549 MiB | 6.9 tok/s |
  | **IQ4_NL-ps4b** | **14.81 ±0.65** | **5,209 MiB** | **8.6 tok/s**（`-fa off`） |

- 中身: IQ4_NL（FFNと注意の行列、3.2GB）、Q6_K（出力ヘッドとattn_v）、Q4_K（token_embd、CPU側）。

### Step 12: MTPによる投機的デコード（今は効果なし、保留）
- `--spec-type draft-mtp --spec-draft-n-max N`（llama-server / llama-cli）。MTP層は本体の重みを共有する。
- 列数の上限を4にした場合の確認結果:

  | 形式 | 3列 | 4列 |
  |---|---|---|
  | IQ4_NL（4行担当）、Q6_K、Q4_K、Q8_0、F32 | 正常 | 正常 |
  | Q5_K | 正常 | 止まる |

  単純に「行数×列数 > 16」で決まるのではなく、シェーダーごとのレジスタの使い方で決まる。なので、作り直すモデルではQ5_Kを使わない。
- IQ4_NLは、同時に計算するトークンを増やしても時間があまり増えない（1トークン 402µs、2トークン 562µs、3トークン 635µs、4トークン 746µs）。
- 計測（Q4_K_M、`scripts/mtp_bench.sh`、生成160トークン、temperature 0）:

  | 設定 | 速度 | 予測が当たった割合 |
  |---|---|---|
  | MTPなし | **6.71 tok/s** | - |
  | 1トークン予測 | 5.55 tok/s | 54% |
  | 2トークン予測 | 3.39 tok/s | 28% |
  | 3トークン予測 | 2.90 tok/s | 22% |

- IQ4_NL-ps4bで測り直した結果（`-fa off`）:

  | 設定 | 速度（英語／日本語） | 予測が当たった割合 |
  |---|---|---|
  | MTPなし | **8.77 / 8.88 tok/s** | - |
  | 1トークン予測 | 6.31 / 5.81 tok/s | 56% / 45% |
  | 2トークン予測 | 5.89 / 5.70 tok/s | 45% / 43% |

- 1回の「予測＋検証」に、MTPなしの1トークンの約1.9倍（約219ms）かかる。2トークン同時の行列計算は1.3倍程度のはずなので、残りは状態を持つ層（線形注意）の状態の巻き戻しなどにかかっていると見ている。
- 今の重さだと、元が取れるのは、予測の当たる割合が1トークン予測で約92%以上、2トークン予測で約75%以上のとき。
- 結論: 出力はMTPなしと完全に一致する（計算は正しい）。ただ、予測の当たる割合が低いうえに、検証の負担が大きいので、かえって遅くなる。公式も「MTP用の重みと設定は後日更新」としているので、更新を待つ。検証時の状態の巻き戻しを軽くする作業も別途必要。

### Step 14: 行列×ベクトル計算の専用シェーダー（失敗、記録として残す）
- 1トークンの内訳（IQ4_NL版、127ms）: IQ4_NLの行列計算が約90ms（71%）、出力ヘッド8.5ms、Flash Attention 7.1ms、RMS正規化5.8ms、その他約15ms。
- IQ4_NL専用のシェーダーを4種類書いたが、どれも汎用（402µs）に勝てなかった（4096×14336、1トークン）。

  | 版 | 考え方 | 1トークン | 2トークン |
  |---|---|---|---|
  | 汎用 | 元から入っているもの | **402 µs** | 562 µs |
  | v1 | 重み2個分をまとめて引くペアの表（256通り） | 425 µs | **467 µs** |
  | v2 | 16通りの表＋次の繰り返しの先読み | 457 µs | 566 µs |
  | v3 | 2ブロック（36バイト）を大きな読み込みで取る | 765 µs | 1147 µs |
  | v4 | 連続した領域を共有メモリにまとめて読み込む | 543 µs | 811 µs |

- 分かったこと:
  - GPUのクロックは800MHzで正常（起動時のログより）。
  - キャッシュに収まる大きさでも速くならない（IQ4_NLで414µs）。メモリ帯域ではなく、GPU内部の命令処理で詰まっている。
  - v3は、各スレッドが飛び飛びの場所を読むので、1命令で触るキャッシュラインが増えて遅くなった。
  - v4から表引きを取り除いても速くならなかった（533µs）。表引きが壁なわけでもない。
  - 本当の原因を突き止めるにはプロファイラ（RGP）が必要。v4は `experiments/` に保存した。
- 結論: 汎用シェーダーのままにする。2トークン同時ならv1が速いので、MTPが良くなったら再検討する。
- 計測用に、キャッシュに収まる行列を64回使い回す計測を `test-backend-ops` に追加した（`m=1024,k=512,nr=[64,1]`）。

### Step 15: routerのカーネル統合
- 元の実装では、mHCのrouterが1トークンの時間の約7割（Q4_K_Mで約360ms）を占めていた。Step 4とStep 8で数msまで減らしたが、まだ4つの処理に分かれていた。
  - RMS正規化（8192要素）→ routerの行列計算（8×8192、F32）→ バイアスの加算 → MHC_ROUTE
- `ggml_mhc_route_ext(x, w, b, eps, post_scale, n_iter)` を追加した（`GGML_OP_MHC_ROUTE` に重みとバイアスを渡す形）。1トークンを1ワークグループで計算する（`vulkan-shaders/mhc_route_fused.comp`）。
  1. 入力8192要素を16バイト単位で読みながら、二乗和と8行分の内積を同時に積み上げる
  2. 256スレッドで集計する
  3. `scale = 1/sqrt(二乗和/K + eps)` をかけてバイアスを足し、sigmoidとSinkhornを計算して書き出す
- 数式は `W·(x/rms) = (W·x)/rms` なので、結果は変わらない（丸めの順番だけ違う）。CPU版も用意し、`test-backend-ops` で一致を確認した（`MHC_ROUTE_EXT`、4ケース）。
- LoRAやHadamard変換付きの重みのときは、従来の経路に戻る。
- 結果: routerの合計時間が約6msから約4.0msになり（1回41µs×96回）、116msから**111〜113ms（8.9〜9.0 tok/s）**になった。
- 品質: perplexityは14.8140（統合前は14.8138）で、変わらない。

### Step 16: 小さな改善の仕上げ
- **Sinkhornのsoftplusを半分に**: `-softplus(d) = -softplus(-d) - d` を使い、1回の正規化でsoftplus（exp＋log）を1回だけ計算する。Vulkan（`mhc_route.comp`、`mhc_route_fused.comp`）とCPU版の両方を直した。数学的に同じ値で、むしろ桁落ちしにくい。約1ms短縮、perplexityは変わらない（14.8139）。
- **出力ヘッドもIQ4_NLに**（`scripts/quantize_iq4nl_c.sh`、`--output-tensor-type iq4_nl`）: 8.5ms → 約6.5ms。perplexityは14.8139 → 14.8366（+0.15%、誤差の範囲）。
- **見送ったもの**: 線形注意の状態の読み書き（GET_ROWSとCPY、計約3.9ms）。1回25µsは2MBの状態を読み書きする実際のデータ転送で、減らすにはGated DeltaNetの演算を「キャッシュの状態をその場で書き換える」形に作り直す必要がある（Mambaには同じ仕組みがある）。
- 結果: **108.3〜108.6ms（9.2 tok/s）**。

| モデル | perplexity | サイズ | 速度 |
|---|---|---|---|
| Q4_K_M（公式） | 14.87 | 5,549 MiB | 6.9 tok/s |
| IQ4_NL-ps4b（出力ヘッドQ6_K） | 14.81 | 5,209 MiB | 9.0 tok/s |
| **IQ4_NL-ps4c（最終版）** | **14.84** | **4,959 MiB** | **9.2 tok/s** |

### Step 13: APIサーバー化とデモ画面
- **自動起動**: ユーザー単位のsystemdサービスにした（`scripts/llama-server.service` → PS4の `~/.config/systemd/user/`）。`loginctl enable-linger user` はsudoなしで通ったので、ログインしなくても起動時に立ち上がる。モデルの読み込みは約76秒。
  - 操作: `systemctl --user status|restart|stop llama-server`、ログは `journalctl --user -u llama-server -f`
- **デモ画面**: `demo/index.html`（PS4の `~/llm/demo/`）。llama-serverの `--path` で配信するので、PCのブラウザで `http://192.168.0.5:8080/` を開くだけ。
  - ゲーム機のホーム画面風: 青いグラデーションに流れる波、横に並ぶタイル（プリセットの質問）、←→とEnterで操作
  - 右側に生成速度（tok/s、サーバーの `timings` の値）、トークンごとの速度の折れ線、生成数、ms/tok、最高値
  - 「思考」ボタンで考える過程の生成を切り替え（初期はオフ）、Escで停止
- **API**: OpenAI互換（`http://192.168.0.5:8080/v1/chat/completions`）。
- **つまずき（503）**: 初期設定のllama-serverは、会話ごとの状態をRAMにためていく（プロンプトのキャッシュ最大8GB、チェックポイント最大32個）。PS4ではRAMの大半をGPUが使っているので、数回の会話でOOMになり、サーバーが落ちて再読み込み中の503が返った。`--cache-ram 0 --ctx-checkpoints 2 -b 512 -ub 256` で解決。6回連続で質問してもメモリは増えず、9.1 tok/sで安定した。
- **ファイアウォール**: PS4のUFWが8080番を止めているので、一度だけ `sudo ufw allow 8080/tcp` が必要。

---

## 3. 実行コマンド（今の推奨設定）

```sh
# PS4上（PS4のGPUでは、キャッシュなしGTT・列数の上限・VRAM優先が自動で有効になる）
cd ~/llm/llama.cpp-sn
GGML_VK_SUBALLOCATION_BLOCK_SIZE=268435456 \
./build/bin/llama-server -m ~/llm/models/SenseNova-Loop-9B-IQ4_NL-ps4c.gguf \
  --no-host -ngl 99 -fa off -c 4096 -t 8 --host 0.0.0.0 --port 8080
```

| 環境変数 / オプション | 役割 |
|---|---|
| `--no-host` | CPU側の重みを、GTTを消費する固定メモリに置かない |
| `--no-op-offload` | CPU側の重みをGPUにコピーして計算させない（重みの一部をCPUに置くときだけ必要） |
| `GGML_VK_SUBALLOCATION_BLOCK_SIZE=268435456` | 確保の単位を256MBにして、VRAMとGTTに細かく詰める |
| `LLAMA_SENSENOVA_MHC_FUSED=0` | 元の実装（融合なし）に戻す。比較用 |
| `GGML_VK_MUL_MAT_VEC_MAX_COLS=N` | まとめて計算する列数の上限（PS4の初期値は2） |
| `GGML_VK_VRAM_FIRST_MB=N` | VRAMに先に置く量 |
| `GGML_VK_DMMV_RM_KQ` / `_RM_IQ` / `_RM_STDQ` / `GGML_VK_DMMV_WG` | 行列×ベクトル計算の調整用 |
| `GGML_VK_PERF_LOGGER=1` | 処理ごとのGPU時間を表示する |

---

## 4. パッチ

`patches/sensenova-loop-ps4.patch`（OpenSenseNova/llama.cpp `24ea7bb` に対する差分、20ファイル）

```sh
git clone https://github.com/OpenSenseNova/llama.cpp && cd llama.cpp
git checkout 24ea7bb78410e0f0f7bbb362e37c60cd9ecfc761
git apply ../patches/sensenova-loop-ps4.patch
```

| 変更 | 主なファイル |
|---|---|
| `GGML_OP_MHC_ROUTE` の追加（CPU版とVulkan版） | `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml-cpu/ops.cpp`, `vulkan-shaders/mhc_route.comp` |
| hc_pre / hc_postをストリーム数2に対応 | `vulkan-shaders/dsv4_hc_{pre,post}.comp`, `ggml-vulkan.cpp` |
| SenseNovaの融合グラフ | `src/models/sensenova-loop.cpp`, `src/models/models.h` |
| PS4のGPU検出（列数の上限、キャッシュなしGTT、VRAM優先、IQの行数） | `ggml-vulkan.cpp`, `ggml-vulkan-buffers.cpp`, `ggml-vulkan-types.h` |
| F32精度指定付きの行列計算をFP16なしのGPUで許可 | `ggml-vulkan.cpp`（supports_op） |
| テストの追加 | `tests/test-backend-ops.cpp` |

これはローカル用のフォークで、本家llama.cppにPRを出すものではない（本家には、AIが書いたコードについて独自のルールがある）。

---

## 5. ハマりどころまとめ

1. **PS4のGPU特有のハング**: 行列×ベクトル計算で「行数×列数 > 16」になると固まる。メモリを増やしても直らない。
2. **RADVはメモリを多めに確保できてしまう**: 確保の時点では通るが、命令を送る段階で `Not enough memory for command submission` やタイムアウトになる。
3. **llama.cppの自動の挙動がメモリを食う**: CPU側の重みもGTTの固定メモリに置く（`--no-host` で回避）。バッチが32以上だとCPU側の重みをGPUにコピーする（`--no-op-offload` で回避）。
4. **メモリの置き場所で速度が4〜6倍変わる**: Garlic-LLMの名前の由来どおり、GPU直結の速い経路を使うのが肝。
5. **精度指定で演算がCPUに回される**: 精度指定付きの小さな行列計算ひとつで、1トークンあたり97回の同期が起きていた。
6. **PS4のGPUでは量子化のbit数が速度を決めない**: 展開の計算が軽い形式（IQ4_NL、Q4_0）が速い。
7. 起動ファイルは暗号化されたUFS2領域にあるので、Linuxからは書き換えない。PS4を通常モードで起動してFTPで書き換える。

---

## 6. 動画用メモ（X向け）

- **見せ場**
  1. PS4（ゲーム機）がAPIサーバーとして応答している様子（PCのチャット画面と、PS4のSSH画面を並べる）
  2. 速度の推移のグラフ: 0.88 → 1.4 → 1.6 → 5.0 → 6.9 → 8.4 → 8.6 → 9.0 → 9.2 tok/s
  8. routerの話: 元の実装では1トークンの7割（約360ms）がrouterだった → 演算の統合で約4msに（約90分の1）
  6. 「PS4ではbit数より展開の軽さで速度が決まる」→ IQ4_NLに作り直して、品質そのままで25%速く
  7. 失敗談: 専用シェーダー4種類が全敗した話（キャッシュに収まっても速くならない）
  3. 融合あり・なしの比較: 同じ出力のまま3.5倍速い（演算数52,749 → 4,161）
  4. 「GarlicとOnion」の話: 同じメモリでも経路で4〜6倍違う
  5. PS4のGPUが固まる条件を突き止めた話（行数×列数 > 16）
- **一言で言うと**: 「2周するループ構造とmHCを持つ9BモデルをPS4のGPUだけで動かし、6.9 tok/sまで速くした」
- **撮るときの注意**
  - PS4はデスクトップなしで起動する（SSHのみ）
  - 最初の1回は重みの読み込みに約1.5分かかる（HDDなので）
  - llama-serverの `timings`（`predicted_per_second`）を画面に表示すると速さが伝わる

---

## 7. TODO
- [x] imatrixの完了 → IQ4_NLに量子化 → perplexityで品質比較 → 速度計測（8.6 tok/s、品質は同じ）
- [x] 列数の上限4の安全確認 → MTPの計測（今は逆効果）
- [x] IQ4_NLの専用シェーダー（4種類とも汎用に勝てず）
- [x] routerのカーネル統合（Step 15）、softplusを半分に、出力ヘッドのIQ4_NL化（Step 16）→ 9.2 tok/s
- [x] MTPをIQ4_NL版で再計測（やはり逆効果）
- [ ] （将来）MTPの公式更新を待つ。Gated DeltaNetの状態をその場で書き換える形にする（最大約2ms）。GPUの計測用シェーダーで「コスト表」を作ってから、行列計算を書き直す
- [ ] llama-serverのsystemd化、PC側のデモ用クライアント
- [ ] 動画の撮影と編集
