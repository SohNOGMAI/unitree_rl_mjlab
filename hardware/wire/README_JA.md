# G1ワイヤー超低速吊上げ試験

`slow_hoist_test.py`はG1制御とは独立した、RobStride06＋GL40_2用の短時間
試験です。既存の`wire_main.py`と同じCAN ID、モータ型、プーリ半径、符号、
GL40バックテンションを使用します。

## 重要な制約

- RS06定格トルクは11 N·mです。半径51 mmでは理想張力約216 Nです。
- G1 XML質量33.34 kgとモジュール2.5 kgの鉛直支持には約352 N、つまり
  約17.9 N·mが必要で、すでに定格を超えます。
- 480 Nは24.48 N·mでピーク領域です。連続運転値ではありません。
- このスクリプトは初回値として5 mm/s、10 mm移動、0.5秒保持を使います。
- 外部ロードセルがないため、`estimated_tension_n`はモータ推定トルクを
  プーリ半径で割った値であり、校正済み実張力ではありません。
- 独立した二次支持を、落下量が10 mm以下になるよう設置してください。
- 物理E-stopと立入禁止範囲が必要です。

## 1. まず計算だけ確認

```bash
cd "$HOME/workspace/unitree_rl_mjlab"

python3 hardware/wire/slow_hoist_test.py
```

この実行ではCANを開かず、モータも有効化しません。

実機実行に使うPython環境へモータライブラリをまだ入れていない場合は、
先に次を一度実行します。

```bash
python3 -m pip install -e \
  "$HOME/workspace/evarl_robot/evarl_robot_common/evarl_motors_lib"
```

## 2. 無荷重で符号を確認

G1を接続せず、ワイヤーにも荷重を掛けない状態で行ってください。今回の
吊上げスクリプトは全荷重用なので、無荷重で直接`--execute`しないで
ください。既存の低トルクモータ試験で、正指令が巻取り方向であることを
確認します。逆の場合は本スクリプトへ`--reel-sign -1`を渡します。

## 3. 短時間吊上げ試験

CANを準備してから実行します。

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

cd "$HOME/workspace/unitree_rl_mjlab"

python3 hardware/wire/slow_hoist_test.py \
  --execute \
  --bus can0 \
  --vertical-component 1.0 \
  --speed-m-s 0.005 \
  --travel-m 0.010 \
  --max-tension-n 480 \
  --csv results/wire_hoist/slow_hoist.csv
```

`--vertical-component`はワイヤー単位方向ベクトルの鉛直成分です。初回は
ワイヤーをほぼ鉛直にして1.0を使います。斜めの場合は実際の幾何から
入力します。必要張力が480 N以上になる値は、コードが起動前に拒否します。

実行中にSpaceを押すと張力を制限速度で落とします。Ctrl-Cは通信可能なら
ゼロ指令後、直ちにモータをdisableします。どちらもPCやCANが故障した場合
には効かないため、物理E-stopの代用ではありません。

この試験はG1の`g1_ctrl`と通信しません。G1は先にStandポリシーへ入れ、
吊上げ開始後の固定姿勢への遷移はUnitree純正コントローラで別に行います。
