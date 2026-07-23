# G1ワイヤーモジュール現地試験手順

この文書は、現地到着後にワイヤーモジュールを単体確認し、条件を満たした
場合だけ短時間のG1吊上げ確認へ進むための手順です。上から順番に実施し、
途中の確認を省略しないでください。

制御対象は次の2台です。

| 役割 | モータ | CAN ID | 既定方向 |
|---|---|---:|---:|
| 主巻取り | RobStride06 | 127 | 正トルク・正速度で巻取り |
| たるみ防止 | GL40_2 | 1 | 正トルクで張力付与 |

`slow_hoist_test.py`はG1の`g1_ctrl`とは独立したプログラムです。G1は
Ethernet、ワイヤーはCAN-USB経由の`can0`を使用するため、同じPCの別
ターミナルで同時に動かせます。

---

## 0. 絶対に守る条件

- 最初はG1をワイヤーへ接続しない。
- 無荷重状態で`slow_hoist_test.py --execute`を既定値のまま実行しない。
  既定値は約35.84 kgを想定し、約352 Nまで指令を上げる。
- CAN配線を変更するときはモータ主電源を切る。
- モータ、プーリ、ワイヤーと同じ平面に人を立たせない。
- 手、髪、服、工具をプーリへ近づけない。
- モータ電源を切れる物理E-stopを担当者が持つ。
- PCのSpaceやCtrl-Cは物理E-stopの代用ではない。
- G1試験時は、ワイヤーが失陥しても落下量10 mm以下で受け止める独立した
  クレーンまたは二次ハーネスを必ず使用する。
- RS06の11 N·mは定格、36 N·mはピークである。半径51 mmで480 Nを出す
  には24.48 N·mが必要で、連続運転はできない。

次のいずれかが発生したら、その段階で中止します。

- CAN状態が`BUS-OFF`、`ERROR-PASSIVE`またはエラーカウンタ増加
- CAN応答が`None`、`No message received`、タイムアウト
- RobStrideのErrorが0以外
- GL40が通常のenable状態以外
- 異音、焦げ臭、発煙、急激な温度上昇
- ワイヤーの毛羽立ち、つぶれ、キンク、プーリからの脱線
- 指令と反対方向への回転
- 予想外の高速回転または振動

---

## 1. 必要な人員と機材

### 人員

最低2人、G1試験では3人を推奨します。

1. PC操作者
2. 物理E-stop担当者
3. G1・クレーン監視者

### 機材

- UbuntuノートPC
- CAN-USBアダプタ
- CAN-H、CAN-L、信号GND配線
- CAN終端抵抗2個（通常120 Ω）
- RobStride06、GL40_2、プーリ、ワイヤー
- モータ電源と物理E-stop
- テスター
- 1 kg程度の既知ダミー荷重
- ダミー荷重用の二次支持
- G1試験時は独立クレーンまたは二次ハーネス

CAN-USBは通信だけに使用します。モータ動力をCAN-USBやPCのUSBから
供給してはいけません。

---

## 2. ノートPC上のコード確認

コードの転送が済んだら確認します。

```bash
cd "$HOME/workspace/unitree_rl_mjlab"

git branch --show-current
git log -3 --oneline
ls -l hardware/wire/slow_hoist_test.py
```

ブランチは`dev/wire`、ファイルは次に存在する必要があります。

```text
~/workspace/unitree_rl_mjlab/hardware/wire/slow_hoist_test.py
```

構文確認を行います。これはモータを動かしません。

```bash
python3 -m py_compile hardware/wire/slow_hoist_test.py
```

何も表示されなければ正常です。

---

## 3. PythonとCANツールの準備

```bash
sudo apt update
sudo apt install -y can-utils python3-pip
```

使用するPython環境を有効にします。

```bash
conda activate unitree_rl_mjlab
```

モータライブラリをインストールします。

```bash
python -m pip install -e \
  "$HOME/workspace/evarl_robot/evarl_robot_common/evarl_motors_lib"
```

確認します。

```bash
python -c "from evarl_motors_lib import create_motor; import can, bitstring; print('motor library OK')"
```

次が出れば正常です。

```text
motor library OK
```

`evarl_robot_common/evarl_motors_lib`がノートPCにない場合は、先にデスクトップ
PCからそのフォルダをコピーします。ROSは今回の単体試験には不要です。

---

## 4. 電源OFF状態の機械・配線確認

モータ主電源とG1の電源を切った状態で行います。

1. ワイヤーに毛羽立ち、キンク、つぶれがない。
2. ワイヤーがプーリ溝に正しく収まっている。
3. プーリ固定ねじ、モータ固定ねじ、アンカー固定部が緩んでいない。
4. CAN-H同士、CAN-L同士、信号GND同士が接続されている。
5. CAN-HとCAN-Lが逆になっていない。
6. 電源線とCAN信号線を取り違えていない。
7. CANバス両端に120 Ω終端がある。

電源OFFでCAN-HとCAN-L間の抵抗を測ります。120 Ωが2個正しく並列に
入っていれば、概ね60 Ωです。大きく異なる場合は電源を入れません。

配線概略：

```text
CAN-USB CAN-H ── RobStride CAN-H ── GL40 CAN-H
CAN-USB CAN-L ── RobStride CAN-L ── GL40 CAN-L
CAN-USB GND   ── RobStride GND   ── GL40 GND
```

---

## 5. CAN-USBの認識確認

CAN-USBを接続して確認します。

```bash
ip -brief link
lsusb
```

`ip -brief link`に次のような表示があれば、SocketCANとして認識されています。

```text
can0             DOWN
```

`can1`など別名なら、以後の`can0`をその名前に読み替えます。

`can0`がなく、`/dev/ttyUSB0`または`/dev/ttyACM0`だけが現れる場合は
SLCAN型の可能性があります。アダプタ固有の設定が必要なので、その状態で
以下のモータ試験へ進まず、次の結果を保存します。

```bash
lsusb
ls -l /dev/ttyUSB* /dev/ttyACM*
dmesg | tail -50
```

---

## 6. can0を1 Mbpsで起動

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
```

確認します。

```bash
ip -details -statistics link show can0
```

次を確認します。

```text
state ERROR-ACTIVE
bitrate 1000000
```

`BUS-OFF`または`ERROR-PASSIVE`なら、配線、終端、GND、ビットレートを
修正するまで進みません。

---

## 7. モータ電源投入

1. ワイヤー端をG1や人へ接続しない。
2. プーリ周辺を立入禁止にする。
3. E-stop担当者が停止スイッチを持つ。
4. PCターミナルを操作可能な状態にする。
5. モータ主電源を投入する。

電源投入直後に勝手に回る、振動する、異音がする場合は、ただちに物理
E-stopで電源を切ります。

CAN状態をもう一度確認します。

```bash
ip -details -statistics link show can0
```

---

## 8. RobStride06ゼロトルク通信試験

この試験はモータをenableしますが、指令トルクは0です。

```bash
python -m evarl_motors_lib.examples.xiaomimotor_test \
  --device can0 \
  --ids 127 \
  --motor_type RobStride06 \
  --task sense \
  --hz 100
```

位置、速度、トルク、温度、Errorが表示されたら、Enterを押して正常終了
させます。

確認項目：

- CAN IDが127
- Position、Velocity、Torque、Tempが数値
- Errorが0
- 停止時に`correctly disabled!`が表示される

`None`やタイムアウトがあれば進みません。

---

## 9. RobStride06低トルク方向試験

ワイヤーをG1へ接続せず、自由端が巻き込まれないよう固定・案内します。
最初は0.2 N·m、理想ケーブル力約3.9 Nだけを指令します。

```bash
python -m evarl_motors_lib.examples.xiaomimotor_test \
  --device can0 \
  --ids 127 \
  --motor_type RobStride06 \
  --task torque \
  --value 0.2 \
  --hz 100
```

1秒以内にEnterを押して停止します。

確認項目：

- 正トルクで主プーリが巻取り方向へ回る
- ワイヤーがプーリ溝から外れない
- 異音、振動、引っかかりがない
- Enter後にモータがdisableされる

正トルクで送り出す場合は記録します。本番スクリプトに
`--reel-sign -1`を付けます。方向が不明なままG1試験へ進みません。

---

## 10. GL40ゼロトルク通信試験

```bash
python -m evarl_motors_lib.examples.tmotor_test \
  --device can0 \
  --ids 1 \
  --motor_type GL40_2 \
  --task sense \
  --hz 100
```

数値が表示されたらEnterで正常終了します。

確認項目：

- CAN IDが1
- Position、Velocity、Torque、Tempが数値
- enable状態として応答する
- 終了時に`Disabling Motors...`が表示される

---

## 11. GL40低トルク方向試験

```bash
python -m evarl_motors_lib.examples.tmotor_test \
  --device can0 \
  --ids 1 \
  --motor_type GL40_2 \
  --task torque \
  --value 0.02 \
  --hz 100
```

1秒以内にEnterを押します。

確認項目：

- 正トルクがたるみを取る方向か
- 異音、振動、脱線がないか
- Enter後にdisableされるか

逆なら、後のスクリプトで`--gl40-dir-command -1`を使用します。
`--gl40-dir`と`--gl40-dir-command`を同時に変更せず、まずcommand側だけを
変更します。

---

## 12. 計算のみモード

モータをdisableし、G1をまだ接続しない状態で実行します。

```bash
cd "$HOME/workspace/unitree_rl_mjlab"

python hardware/wire/slow_hoist_test.py
```

これはCANを開かず、モータを動かしません。既定の計算結果は概ね次です。

```text
total suspended mass : 35.841 kg
weight               : 351.6 N
support feed-forward : 351.6 N
cable speed          : 5.0 mm/s
test travel          : 10.0 mm
hard tension limit   : 480.0 N (24.48 Nm)
RS06 rated force     : 215.7 N (11 Nm)
minimum usable u_z   : 0.733
peak-region timeout  : 8.0 s
```

---

## 13. 1 kgダミー荷重で統合試験

個別モータ試験に成功した後、G1ではなく約1 kgの既知荷重で主巻取りと
GL40を同時に確認します。

1. ダミー荷重へ独立した二次支持を付ける。
2. 落下可能距離を5 mm以下にする。
3. ワイヤーをほぼ鉛直にする。
4. 人をワイヤー延長線と荷重の下から退避させる。
5. E-stop担当者を配置する。

実行：

```bash
cd "$HOME/workspace/unitree_rl_mjlab"

python hardware/wire/slow_hoist_test.py \
  --execute \
  --bus can0 \
  --robot-mass-kg 1.0 \
  --module-mass-kg 0.0 \
  --vertical-component 1.0 \
  --speed-m-s 0.002 \
  --travel-m 0.005 \
  --max-tension-n 30 \
  --hold-seconds 0.5 \
  --csv results/wire_hoist/dummy_1kg.csv
```

方向確認で反転が必要だった場合は、確認済みのオプションを追加します。

```text
--reel-sign -1
```

またはGL40だけなら、

```text
--gl40-dir-command -1
```

実行前に次を正確に入力します。

```text
SECONDARY SUPPORT READY
```

期待するシーケンス：

```text
RAMP_UP -> HOIST -> HOLD -> RAMP_DOWN -> DISABLE
```

期待値：

- `reel_in_m`が巻取り正方向に約+0.005 m
- `cable_velocity_m_s`が概ね+0.002 m/s
- 指令張力が30 Nを超えない
- 終了後にRobStrideとGL40がdisable
- CSVが保存される

不安定、逆回転、急加速ならワイヤーターミナルでSpaceを押します。PCやCANが
反応しなければ物理E-stopを使用します。

---

## 14. ダミー試験後の確認

CAN統計：

```bash
ip -details -statistics link show can0
```

CSV：

```bash
head -5 results/wire_hoist/dummy_1kg.csv
tail -5 results/wire_hoist/dummy_1kg.csv
```

主なCSV列：

- `state`
- `reel_in_m`
- `cable_velocity_m_s`
- `velocity_ref_m_s`
- `tension_cmd_n`
- `estimated_tension_n`
- RobStride位置、速度、トルク、温度、status
- GL40速度、トルク、温度、status

次のすべてを満たした場合だけG1試験を検討します。

- CANエラー増加なし
- 両モータの方向が確定
- 5 mm移動量で自動停止
- 張力指令上限が機能
- SpaceでRAMP_DOWNへ移行
- 両モータが最後にdisable
- ワイヤー・プーリ・固定部に損傷なし

---

## 15. G1短時間10 mm吊上げ試験の前提条件

この章は単体・ダミー試験にすべて合格した場合だけ実施します。

### 機械条件

- G1本体は独立クレーンまたは二次ハーネスで支持する。
- ワイヤー失陥時の落下量は10 mm以下。
- ワイヤーは初回にはほぼ鉛直、`u_z ≈ 1.0`。
- 人はG1を手で受け止めない。
- モータ、プーリ、ワイヤー、アンカー、背中取付部の全てが480 N以上の
  使用荷重に対応していることを担当者が確認する。

### 幾何条件

ワイヤー単位方向の鉛直成分は、

```text
u_z = delta_z / sqrt(delta_x^2 + delta_y^2 + delta_z^2)
```

です。今回の質量では`u_z < 0.733`になる配置は480 Nでも支持できないため、
コードが実行を拒否します。上限を増やさず、アンカーを鉛直側へ移します。

### G1ターミナル

```bash
cd "$HOME/workspace/unitree_rl_mjlab"

./deploy/robots/g1/build/g1_ctrl \
  --network enp1s0f0 \
  --domain-id 0 \
  --log
```

G1ターミナルで`1`を一度押し、Standへ移行します。10～20秒待ち、足踏み、
yaw回転、振動、腰折れがないことを確認します。

### ワイヤーターミナル

```bash
cd "$HOME/workspace/unitree_rl_mjlab"
conda activate unitree_rl_mjlab

python hardware/wire/slow_hoist_test.py \
  --execute \
  --bus can0 \
  --vertical-component 1.0 \
  --speed-m-s 0.005 \
  --travel-m 0.010 \
  --max-tension-n 480 \
  --csv results/wire_hoist/g1_10mm.csv
```

全員の準備完了後だけ、

```text
SECONDARY SUPPORT READY
```

を入力します。

この10 mm試験ではFixedPostureへ変更しません。巻取り、短時間保持、張力低下、
disableが正常に完了することだけを確認します。

---

## 16. 操作と停止対象

| 操作場所 | 操作 | 対象と動作 |
|---|---|---|
| G1ターミナル | `1` | G1をStandへ遷移 |
| G1ターミナル | `Shift+F` | 完全支持後だけFixedPostureへ遷移 |
| G1ターミナル | `Space` | G1をPassiveへ遷移 |
| ワイヤーターミナル | `Space` | ワイヤー張力を制限速度で低下 |
| ワイヤーターミナル | `Ctrl-C` | ゼロ指令を試行後、モータdisable |
| 物理E-stop | 電源遮断 | PC/CANが応答しない場合の停止 |

G1ターミナルのSpaceはワイヤーを止めません。ワイヤーターミナルのSpaceは
G1を止めません。ターミナルのフォーカス先を必ず確認します。

---

## 17. 試験終了手順

1. ワイヤースクリプトが`DISABLE`まで完了したことを確認する。
2. 二次支持または床が荷重を受けていることを確認する。
3. G1試験ならG1ターミナルでSpaceを押してPassiveへ戻す。
4. G1を独立クレーンで完全支持する。
5. ワイヤーモータ主電源を切る。
6. G1を正規手順で停止する。
7. CANインターフェースを停止する。

```bash
sudo ip link set can0 down
```

8. 電源OFFを確認してからワイヤーとCAN配線を外す。
9. ワイヤー、プーリ、固定部、モータ温度を点検する。
10. CSVとG1ログを保存する。

```bash
cp deploy/robots/g1/log/log.txt \
  "$HOME/g1_real_test_$(date +%Y%m%d_%H%M%S).log"
```

---

## 18. 現地で記録する内容

各試験について次をメモします。

- 日時
- CAN-USB機種
- CANインターフェース名
- RobStride IDとGL40 ID
- RobStride、GL40の方向設定
- 実効プーリ半径
- 試験荷重
- `vertical-component`
- 指令速度、移動量、張力上限
- 最大推定張力
- 最大温度
- CANエラー数
- 成功／中止と理由
- CSVファイル名
- G1ログファイル名

値が不明な項目を推測で埋めず、不明と記録します。
