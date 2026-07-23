# G1実機：歩行・しゃがみ・固定姿勢の初学者向け手順

## 1. 何を実装したか

このフォルダの `g1_ctrl` は、G1から関節角度とIMU情報を受け取り、
50 Hzでニューラルネットワークを計算し、29個の関節目標角をG1へ送り
ます。MuJoCoを動かす `scripts/play.py` は実機では使用しません。

実機コントローラには次の状態があります。

1. `Passive`: 関節を強く位置保持せず、ダンピングだけをかけます。
2. `FixStand`: 現在姿勢から通常の直立姿勢へ2秒で移動します。
3. `Velocity`: 歩行ポリシーです。左スティックで歩行速度を指令します。
4. `Crouch`: 静止歩行ポリシーからしゃがみポリシーへ切り替え、4秒で
   しゃがみ深さ0.90へ移動し、その姿勢を保持します。
5. `FixedPosture`: ascendで使用した `wide_lift` 固定姿勢です。これは
   自立用の姿勢ではないので、安全ハーネスで支持されている場合だけ使用
   します。

ワイヤーモータ制御はまだこのプログラムに含まれていません。

## 2. Unitree純正無線コントローラの操作

実機版は、G1の `LowState` に含まれるUnitree純正無線コントローラ入力を
使用します。PCへUSB接続したPS4コントローラは読みません。以下の
`A/B/X/Y` はUnitreeコントローラ上の表記です。

| 現在の状態 | 操作 | 次の状態 |
|---|---|---|
| Passive | LT + 十字上 | Stand（速度指令を厳密に0へ固定した歩行ポリシー） |
| Passive | LT + X | FixStand（関節姿勢試験用） |
| FixStand | RT + A | Velocity（歩行） |
| Stand | RT + A | Velocity（スティック歩行） |
| Stand | RT + B | Crouch |
| Velocity | RT + B | Crouch |
| Crouch | RT + X | FixStand |
| Crouch | RT + Y | FixedPosture（支持中のみ） |
| Stand | RB + Y | FixedPosture（支持中のみ） |
| FixedPosture | RT + X | FixStand |
| Passive以外 | LT + B | Passive（ソフトウェア停止） |

たとえばStand開始は、Unitreeコントローラの`LT`を先に保持し、そのまま
十字上を一度押します。

また、`g1_ctrl`を実行しているPCターミナルでSpaceキーを押すと、現在の
状態にかかわらず`Passive`への遷移を要求します。ターミナルに入力
フォーカスがあり、PC・Ethernet・プロセスが正常な場合だけ動作する
ソフトウェア停止です。通信断やPC停止に備える独立した物理E-stopの代用には
なりません。

Unitree純正コントローラを使わず、`g1_ctrl`を実行しているPCターミナル
から状態を変更することもできます。

| 現在の状態 | PCキー | 次の状態 |
|---|---|---|
| Passive | `1` | Stand（速度0の立位ポリシー） |
| Passive | `4` | FixStand（姿勢試験用） |
| Stand | `2` | Crouch |
| Stand | `5` | Velocity |
| Stand（完全支持後） | `Shift+F` | FixedPosture |
| Velocity | `2` | Crouch |
| Crouch | `4` | FixStand |
| Crouch（完全支持後） | `Shift+F` | FixedPosture |
| FixedPosture | `4` | FixStand |
| Passive以外 | `Space` | Passive |

`F`は大文字なので、Shiftを押しながら`F`を押します。FixedPostureは
自立制御ではありません。ワイヤーまたは独立ハーネスが機体荷重を受けた
ことを目視確認してからだけ使用します。キーは`g1_ctrl`のターミナルへ
フォーカスがある場合だけ有効です。

深くしゃがんだ状態から歩行ポリシーへ直接切り替えると、目標角が急変して
転倒しやすくなります。そのため `Crouch -> FixStand -> Velocity` の順で
戻します。

## 3. Crouch内部で行っている処理

しゃがみ開始からの処理は次の順番です。

1. 状態に入った瞬間の29関節の実測角を最初の指令にします。これにより、
   状態遷移した瞬間の角度ジャンプを防ぎます。
2. 最初の1秒で、速度0の歩行ポリシー出力からしゃがみポリシー出力へ
   滑らかに混合します。この1秒間、しゃがみ深さはまだ0です。
3. 次の4秒で、しゃがみ深さを0から0.90へ増やします。
4. 深さに応じた基準関節角に、しゃがみポリシーが出すバランス補正を加え
   ます。
5. hip rollとankle rollの補正を深さに応じて弱め、waist yawの補正は常に
   0にします。これは学習時と同じ処理です。
6. 1制御周期の目標角変化を制限し、ポリシー切替による急な動きを抑えます。

ニューラルネットワークの入力は98個、出力は29個です。主な入力は、

- G1内蔵IMUの角速度
- G1内蔵IMUから計算した重力方向
- しゃがみ深さ
- しゃがみ開始時からのyaw誤差
- 29関節の角度と速度
- 前回の29個のポリシー出力

です。

シミュレーション学習時には前後位置誤差と胴体の平面速度も使用しました。
しかし、標準のG1 `LowState` からは、内蔵運動制御を停止した状態で信頼でき
る世界座標位置・並進速度を取得できません。今回の外部センサなし実装では、
この3値を0として入力します。yawは内蔵IMUから取得するため0固定ではあり
ません。この差は実機試験で最初に確認すべきsim-to-real差です。

## 4. 使用しているモデル

歩行：

```text
logs/rsl_rl/g1_velocity/2026-06-20_00-41-59/model_10000.pt
```

実機用ONNX：

```text
deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx
```

しゃがみ：

```text
logs/rsl_rl/g1_crouch/2026-07-19_11-51-25_gap_planted_no_yaw_s24/model_675.pt
```

実機用ONNX：

```text
deploy/robots/g1/config/policy/crouch/v0/exported/policy.onnx
```

`.pt` はPyTorchの学習チェックポイントです。C++実機コントローラは `.pt`
を直接読めないため、`scripts/export_actor_onnx.py` で `.onnx` へ変換します。

再変換する場合：

```bash
cd /home/soonogami/workspace/unitree_rl_mjlab
conda activate unitree_rl_mjlab

python scripts/export_actor_onnx.py \
  logs/rsl_rl/g1_crouch/2026-07-19_11-51-25_gap_planted_no_yaw_s24/model_675.pt \
  deploy/robots/g1/config/policy/crouch/v0/exported/policy.onnx
```

## 5. 必要なもの

- 29DoF低レベル制御に対応したUnitree G1
- Unitree無線コントローラ
- Ubuntu PC（GPU不要）
- Ethernetケーブル
- G1を支持できる安全ハーネス
- G1の転倒・周囲接触を監視する補助者

最初の試験ではワイヤーモジュールを取り付けず、電源も入れません。

## 6. PCのソフトウェア準備

ビルドとは、人が読めるC++コードをPCが実行できる `g1_ctrl` に変換する
作業です。次の道具をインストールします。

```bash
sudo apt update
sudo apt install -y \
  cmake g++ build-essential git \
  libyaml-cpp-dev libeigen3-dev libboost-all-dev \
  libspdlog-dev libfmt-dev zlib1g-dev
```

次に、G1と通信するUnitree SDK2をインストールします。

```bash
cd /home/soonogami/workspace
git clone https://github.com/unitreerobotics/unitree_sdk2.git

cmake \
  -S unitree_sdk2 \
  -B unitree_sdk2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local

cmake --build unitree_sdk2/build -j4
sudo cmake --install unitree_sdk2/build
sudo ldconfig
```

`git clone` でSDKのソースを取得し、`cmake -S ... -B ...` でビルド設定を
作り、`cmake --build` でコンパイルし、`cmake --install` でシステムへ
配置します。

## 7. 実機コントローラをビルドする

```bash
cd /home/soonogami/workspace/unitree_rl_mjlab

cmake \
  -S deploy/robots/g1 \
  -B deploy/robots/g1/build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build deploy/robots/g1/build -j4
```

成功すると次の実行ファイルができます。

```text
deploy/robots/g1/build/g1_ctrl
```

不足ライブラリがないか確認します。

```bash
ldd deploy/robots/g1/build/g1_ctrl | grep "not found"
```

何も表示されなければ正常です。

## 8. まずunitree_mujocoで試す

この試験中は実機G1へのEthernetを抜いてください。

```bash
cd /home/soonogami/workspace/unitree_rl_mjlab

cmake -S simulate -B simulate/build -DCMAKE_BUILD_TYPE=Release
cmake --build simulate/build -j4
./simulate/build/unitree_mujoco
```

MuJoCoは安全のため一時停止状態で起動します。この時点では画面内のG1が
静止していて正常です。物理計算を開始する前に、次の端末でコントローラを
起動してください。

別のターミナルを開きます。

```bash
cd /home/soonogami/workspace/unitree_rl_mjlab
./deploy/robots/g1/build/g1_ctrl --network=lo --log
```

`lo` はPC内部だけで通信する仮想ネットワークです。実機へは送信しません。
`Connected to robot` と `FSM: Start Passive` が表示されたら、PS4コントローラーで
`L2 + 十字上` を押し、速度0のStandポリシーへ直接切り替えます。次のログを
確認してからMuJoCoウィンドウでSpaceキーを押し、物理計算を開始します。

```text
FSM: Change state from Passive to Stand
```

`FixStand`は関節角を補間するだけで空中の胴体姿勢をフィードバックしないため、
起動経路には使用しません。Standポリシーで安定して直立したことを確認してから、
`R2 + ○` でしゃがみポリシーへ移ります。

上記のボタン順に、Stand、Crouchを確認します。
`FixedPosture` は仮想ハーネスがない状態では倒れる可能性があるため、ここで
地面上の自立姿勢として評価しないでください。

リセットする場合は両プログラムを終了して再起動してください。起動後、
`L2 + 十字上`でStandへ移り、そのログを確認してからSpaceキーを押します。

## 9. 実機Ethernetを設定する

接続名を確認します。

```bash
ip -brief link
```

例えばG1用Ethernetが `enp5s0` の場合：

```bash
sudo ip link set enp5s0 up
sudo ip addr add 192.168.123.222/24 dev enp5s0
ip -4 addr show dev enp5s0
```

別の名前なら、以下の全ての `enp5s0` をその名前へ置き換えます。

## 10. 実機の最初の試験

### 10.1 吊り下げ状態

1. ワイヤーモジュールを外します。
2. G1を安全ハーネスで支持し、足を床から少し浮かせます。
3. コントローラのスティックを中央にします。
4. G1を起動してzero-torque状態を待ちます。
5. L2 + R2でdebug modeにします。
6. PCで次を実行します。

```bash
cd /home/soonogami/workspace/unitree_rl_mjlab
./deploy/robots/g1/build/g1_ctrl --network=enp5s0 --log
```

7. `Connected to robot` と `FSM: Start Passive` を確認します。
8. 実機では安全ハーネスで支持し、L2 + □でFixStandへ移動します。
9. 直立が安定してからR2 + ×でVelocityへ移動します。
10. 関節方向や左右の対応に異常がないことを確認します。
11. スティック中央のままVelocityを確認します。
12. R2 + ○でCrouchへ入り、ログが
    `POLICY_BLEND -> CROUCHING -> HOLD` になることを確認します。
13. R2 + □でFixStandへ戻します。

### 10.2 床上、ハーネスあり

吊り下げ試験で異常がなかった場合だけ、足裏へ徐々に荷重を移します。

1. FixStandで直立できるか確認します。
2. Velocityへ入り、速度0で10秒立てるか確認します。
3. 左スティックをわずかに前へ倒して低速歩行を確認します。
4. いったん停止し、速度0で安定してからCrouchへ入ります。
5. Crouchを10秒以上保持できるか確認します。
6. R2 + XでFixStandへ戻します。

固定姿勢は自立姿勢ではないため、床上では押さないでください。

## 11. 主な調整パラメータ

`config/config.yaml` の `FSM.Crouch` にあります。

- `policy_blend_time_s`: 歩行actorからしゃがみactorへ切り替える時間。
- `crouch_duration_s`: 深さ0から目標深さまで移動する時間。
- `target_depth`: しゃがみ深さ。学習範囲内の0～0.90で使用します。
- `max_joint_target_rate_rad_s`: 1秒当たりの関節目標角変化上限。
- `crouch_target_q`: 29関節の最終基準姿勢。

最初の実機試験では値を変更しないでください。動作確認後、一度に1項目だけ
変更し、変更前後のログと動画を残します。

## 12. 重要な制限

- 現在のコードにワイヤーモータ制御はありません。
- `FixedPosture` はバランス制御ではなく、PDで29関節角を保持する状態です。
- 外部位置・並進速度を使わないため、しゃがみ中の大きな前後移動をモデルが
  シミュレーションと同じ方法では補正できません。
- `L2 + B` はPassiveへのソフトウェア遷移で、電源を物理的に遮断する非常
  停止ではありません。また、空中で支持力を生む機能でもありません。常に
  独立した非常停止手段と機械的な安全ハーネスを使用してください。
