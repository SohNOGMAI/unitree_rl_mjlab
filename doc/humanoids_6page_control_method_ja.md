# IEEE-RAS Humanoids 6ページ論文向け制御モデル

## Wire-Assisted Humanoid TraversalのMethod節ドラフト

本資料は、既存の包括資料よりも論文本文へ直接移しやすい粒度で、制御モデルを学術的に整理したものである。6ページ論文では実装の全分岐を列挙せず、共通の数理モデル、ハイブリッド状態機械、モード差、学習済み姿勢遷移の必要性に絞る。

---

## 1. 論文での中心的な定式化

本研究の制御系は、次の二つのサブシステムを上位ハイブリッド制御器で同期する。

1. **Humanoid skill controller**: 歩行、しゃがみ、吊り下げ固定姿勢、着地後立位回復
2. **Cable controller**: 幾何計画、ワイヤ長軌道、張力制御、巻き下ろし、解放

研究の主張は「歩行ポリシーだけで全動作を行う」ことではなく、次のように置く。

> A hierarchical controller coordinates pretrained humanoid motor skills and an independent geometry-based cable controller to extend traversable terrain beyond the robot's unaided locomotion capability.

環境形状とアンカー位置は事前に既知とし、実行中に未知障害物を認識する問題は扱わない。

---

## 2. 仮定

論文冒頭で仮定を明示する。

1. 障害物の境界、支持面高さ、gap幅は既知である。
2. アンカーは世界座標系に固定され、その位置 \(\mathbf p_a\in\mathbb R^3\) は既知である。
3. ワイヤは質量なし、引張専用であり、押力を発生しない。
4. 背部取付点は胴体へ剛に固定される。
5. 現段階では巻取モジュールCADの質量・慣性・弾性を無視し、理想的な外部ワイヤwrenchとしてモデル化する。
6. 主移動は世界座標系の \(x-z\) 平面内で起こるが、ロボットと制御は3次元で計算する。
7. ロボットの固有受容状態とIMUは利用可能である。
8. シミュレーションでは背部位置を直接取得する。実機では運動学・IMU・odometry・winch encoderによる推定へ置き換える。

仮定5により、本研究は制御原理の検証であり、実機モジュールを含む完全な力学検証ではない。

---

## 3. 浮遊ベースロボットとワイヤの力学

### 3.1 ロボット方程式

浮遊ベースを含む一般化座標を \(\mathbf q\)、速度を \(\dot{\mathbf q}\) とする。接触を含むロボットの運動方程式は

\[
\mathbf M(\mathbf q)\ddot{\mathbf q}
+\mathbf h(\mathbf q,\dot{\mathbf q})
=\mathbf S^T\boldsymbol\tau
+\mathbf J_c^T\boldsymbol\lambda_c
+\mathbf J_b^T\mathbf F_w,
\tag{1}
\]

で表す。

- \(\mathbf M\): 慣性行列
- \(\mathbf h\): コリオリ、遠心、重力項
- \(\boldsymbol\tau\): 関節アクチュエータトルク
- \(\boldsymbol\lambda_c\): 足・障害物との接触力
- \(\mathbf J_c\): 接触Jacobian
- \(\mathbf J_b\): 背部取付点の並進Jacobian
- \(\mathbf F_w\): ワイヤ力

論文では式(1)を使うと、ワイヤ力がロボット関節制御と独立した外力として入ることを明確に表現できる。

### 3.2 背部取付点

胴体座標系での取付点を \(\mathbf r_b^B\)、胴体位置と姿勢を \(\mathbf p_t,\mathbf R_{WB}\) とすると、世界座標の取付点は

\[
\mathbf p_b
=\mathbf p_t+\mathbf R_{WB}\mathbf r_b^B.
\tag{2}
\]

実験設定は

\[
\mathbf r_b^B=(-0.14,0,0.40)^T\ \mathrm{m}
\tag{3}
\]

である。

### 3.3 ワイヤ幾何

\[
\mathbf c=\mathbf p_a-\mathbf p_b,
\qquad
L=\|\mathbf c\|,
\qquad
\hat{\mathbf u}=\frac{\mathbf c}{L}.
\tag{4}
\]

引張専用条件より

\[
T\ge0,
\qquad
\mathbf F_w=T\hat{\mathbf u}.
\tag{5}
\]

MuJoCoでは胴体重心へ外部wrenchを与えるため、背部作用点と等価なモーメント

\[
\boldsymbol\tau_w
=(\mathbf p_b-\mathbf p_{com,t})\times\mathbf F_w
\tag{6}
\]

を併せて適用する。

式(6)は本研究で重要である。取付点が胴体重心の後上方にあるため、張力は並進支持だけでなくpitch、roll、yaw運動も励起する。吊り下げ固定姿勢としゃがみ姿勢は、この作用点モーメント下で姿勢を安定させるために導入される。

---

## 4. 幾何ベースのワイヤ長計画

### 4.1 共通計画

各動作モード \(m\in\{\mathrm{asc},\mathrm{des},\mathrm{gap}\}\) に対し、既知の環境幾何 \(\mathcal G\) と安全margin \(\boldsymbol\mu_m\) から、吊り上げ時と着地補助時の目標背部位置を生成する。

\[
\mathbf p_{b,H}^{m}
=g_H^m(\mathcal G,\mathbf p_b^0,\boldsymbol\mu_m),
\tag{7}
\]

\[
\mathbf p_{b,R}^{m}
=g_R^m(\mathcal G,\mathbf p_b^0,\boldsymbol\mu_m).
\tag{8}
\]

ワイヤ長目標は

\[
L_H^m
=\|\mathbf p_a-\mathbf p_{b,H}^{m}\|+\delta_H^m,
\tag{9}
\]

\[
L_R^m
=\max\left(
\|\mathbf p_a-\mathbf p_{b,R}^{m}\|+\delta_R^m,
L_H^m
\right),
\tag{10}
\]

とする。\(\delta_H^m,\delta_R^m\) はワイヤ経路、取付構造、モデル誤差を補正するcalibration marginである。これらを隠れた調整値にせず、実験パラメータとして報告する。

動作開始長を

\[
L_0=\|\mathbf p_a-\mathbf p_b^0\|
\tag{11}
\]

とすると、計画出力は

\[
\Delta L_{reel}=L_0-L_H^m,
\qquad
\Delta L_{pay}=L_R^m-L_H^m.
\tag{12}
\]

したがって、reel-in量とpayout量は操作者が直接指定する入力ではなく、幾何計画の出力である。

### 4.2 モード別目標点

開始支持面高さを \(z_s\)、着地支持面高さを \(z_l\)、開始時の支持面から背部までの高さを

\[
h_b=p_{b,z}^{0}-z_s
\tag{13}
\]

とする。

#### Ascend

段差上面を \(z_l=z_{top}\)、段差高を \(h_s=z_l-z_s\) として

\[
\mathbf p_{b,H}^{asc}
=\mathbf p_b^0+[0,0,h_s+c_{asc}]^T,
\tag{14}
\]

\[
p_{b,R,z}^{asc}=z_l+h_b+m_{asc}.
\tag{15}
\]

\(c_{asc}\)は段差角を越えるclearance、\(m_{asc}\)は着地時に完全解放する前の支持marginである。

#### Descend

降下では大きく巻き上げず、まずslackを除く。

\[
L_H^{des}=L_0-\Delta L_{pre}.
\tag{16}
\]

着地目標点は

\[
\mathbf p_{b,R}^{des}
=\begin{bmatrix}
x_l & y_l & z_l+h_b
\end{bmatrix}^T,
\tag{17}
\]

とする。\((x_l,y_l)\)は既知の着地領域内に設定する。

#### Gap

gapでは手前縁 \(x_f\) と向こう岸縁 \(x_r\) が既知である。向こう岸側アンカーを用い、吊り上げ目標点と着地点を

\[
\mathbf p_{b,H}^{gap}
=\begin{bmatrix}x_f+\epsilon_x&y_0&z_H\end{bmatrix}^T,
\tag{18}
\]

\[
\mathbf p_{b,R}^{gap}
=\begin{bmatrix}x_r+d_l&y_0&z_l+h_b\end{bmatrix}^T
\tag{19}
\]

として計画できる。\(d_l>0\)は向こう岸内側の着地marginである。

現在のコードでは、ascendは式(9)に近い3次元計算を行う一方、descendとgapの一部は鉛直距離近似を使い、gapには経験的な0.30 m/0.20 m offsetが残る。論文で式(7)–(19)を提案法として用いるなら、最終実験コードも同じ3次元目標点計算へ揃えるか、現行近似をImplementation Detailsで明記する必要がある。

---

## 5. 指令長軌道

### 5.1 目標近傍で停止可能な速度則

現在指令長を \(L_c\)、目標を \(L_g\) とし、残距離

\[
d_L=|L_g-L_c|
\tag{20}
\]

を定義する。指令速度は

\[
v_c
=\operatorname{sgn}(L_g-L_c)
\min\left(v_{max},\sqrt{2a_b d_L}\right),
\tag{21}
\]

\[
L_c(k+1)=L_c(k)+v_c\Delta t
\tag{22}
\]

とする。式(21)は停止距離関係から、目標近傍で速度を低下させる。

厳密には \(a_b\) は速度の時間差分を直接拘束する加速度上限ではなく、braking profile parameterである。実機では

\[
|v_c(k+1)-v_c(k)|\le a_{max}\Delta t
\tag{23}
\]

とjerk制限も追加することが望ましい。

LIFTでは \(L_g=L_H\)、LOWERでは \(L_g=L_R\) とする。HOLDでは \(L_c=L_H\)、ASSISTとRELEASEでは \(L_c=L_R\)を保持する。

---

## 6. 長さ追従型張力制御

### 6.1 誤差

\[
e_L=L-L_c,
\qquad
e_v=\dot L_f-\dot L_c.
\tag{24}
\]

差分で得た長さ速度は一次低域filter

\[
\dot L_f(k)
=(1-\alpha)\dot L_f(k-1)+\alpha\dot L_{raw}(k)
\tag{25}
\]

で平滑化する。

### 6.2 準静的吊り上げ・巻き下ろし

\(u_z=\hat{\mathbf u}^T\mathbf e_z\)をワイヤ方向の鉛直成分とする。ロボット重量をワイヤ鉛直成分で支持するfeed-forwardは

\[
T_{ff}^{lift}=\frac{m_tg}{\max(u_z,\varepsilon)}
\tag{26}
\]

である。\(m_t\)には最終的にG1と背負い式モジュールの合計質量を用いる。現シミュレーションではG1質量のみである。

張力目標は

\[
T_d
=\left[
T_{ff}^{lift}+K_pe_L+K_de_v
\right]_+,
\tag{27}
\]

とする。\([x]_+=\max(x,0)\)はワイヤが押せないことを表す。

符号付き誤差を使うことが重要である。正のextensionだけを使うと、ロボットが指令長の内側へ入るたびに張力が0になり、再び落下した後に最大張力が入るため、0–最大張力のlimit cycleが発生する。

### 6.3 Gap振り子中

gapでは鉛直重量を常時完全相殺すると、遠方アンカーに向かう過大水平力が発生する。振り子の半径方向力を用いて

\[
T_{ff}^{swing}
=m_tg u_z+m_t\frac{v_t^2}{L},
\tag{28}
\]

\[
T_d
=\left[
T_{ff}^{swing}+K_{p,g}e_L+K_{d,g}e_v
\right]_+,
\tag{29}
\]

とする。\(v_t\)は取付点速度のワイヤ接線成分である。

現行実装は式(28)の向心力項 \(m_tv_t^2/L\) を陽に入れず、\(m_tgu_z\)と高ゲインPDで補っている。論文では、実際に使った制御を報告するなら向心項なしの式を記載する。より厳密なモデルとして式(28)を採用するなら、コードへ追加して再評価する。

### 6.4 接地中のfall arrester

gapのSETTLE中、遠方アンカーへ張力を出すと足が縁方向へ引かれる。このため、ワイヤはslack margin \(s\) を持つ片側支持として

\[
T_d^{set}
=\left[
K_p(e_L-s)+K_d e_v
\right]_+
\tag{30}
\]

とする。

水平力上限 \(F_{h,max}\) により

\[
T_d^{set}
\le
\frac{F_{h,max}}
{\max(\|\hat{\mathbf u}_{xy}\|,\varepsilon)}
\tag{31}
\]

を課す。これにより、ワイヤは接地姿勢を積極的に前へ引く装置ではなく、倒れ始めた場合だけ支持するfall arresterとなる。

### 6.5 張力制約

\[
\bar T_d=\operatorname{clip}(T_d,0,T_{max}),
\tag{32}
\]

\[
T(k+1)
=T(k)+
\operatorname{clip}
\left(
\bar T_d-T(k),
-\dot T_{max}\Delta t,
\dot T_{max}\Delta t
\right).
\tag{33}
\]

RELEASE時には小さい \(\dot T_{release}\) を使い、急激な荷重移行を防ぐ。

---

## 7. ハイブリッド状態機械

離散状態を

\[
\sigma\in
\{A,S,L,H,D,B,R,F\}
\tag{34}
\]

とする。

| 記号 | 実装名 | ロボット側 | ワイヤ側 | 遷移guard |
|---|---|---|---|---|
| A | APPROACH | +x歩行 | slack | 既知の停止位置へ到達 |
| S | SETTLE | 立位→しゃがみ | slack/fall arrester | 姿勢・速度が連続時間安定 |
| L | LIFT | しゃがみ→固定姿勢 | (L_H)へ巻取 | 長さ・速度・位置が整定 |
| H | HOLD | 固定姿勢 | (L_H)保持 | hold時間経過 |
| D | LOWER | 固定姿勢 | (L_R)へ巻下 | 指令長到達 |
| B | ASSIST | 立位へblend | (L_R)保持 | blend＋支持時間完了 |
| R | RELEASE | 立位 | 張力を徐減 | 張力閾値以下を保持 |
| F | DONE | 歩行再開 | slack | 終了 |

### 7.1 SETTLEからLIFTへのguard

gapではしゃがみ深度到達だけではなく、時間窓 \(\tau_s\) にわたり

\[
|\dot\psi|<\omega_{max},
\quad
|\phi_t|<\phi_{max},
\quad
|v_{b,z}|<v_{z,max},
\quad
|\dot L|<v_{L,max}
\tag{35}
\]

を満たした場合だけLIFTへ移る。

### 7.2 LIFTからHOLDへのguard

\[
|L-L_H|<\epsilon_L,
\quad
|\dot L|<\epsilon_v,
\quad
L_c=L_H.
\tag{36}
\]

gapではさらに

\[
p_{b,x}\ge x_r-\epsilon_x
\tag{37}
\]

を要求し、向こう岸へ到達する前の降下を防ぐ。

### 7.3 実機でのguard置換

式(37)は現行シミュレーションで世界位置を使用する。ワイヤ長・張力中心の実機制御では、足接触、odometry、振り子位相、長さ極値、事前同定した時間などへ置き換える必要がある。この点はLimitationsに記載する。

---

## 8. Humanoid skill controller

### 8.1 スキル集合

\[
\Pi=
\{\pi_{walk},\pi_{crouch},\pi_{susp}^{m}\}.
\tag{38}
\]

- \(\pi_{walk}\): 29 DoF歩行・立位ポリシー
- \(\pi_{crouch}\): 29 DoF接地維持しゃがみポリシー
- \(\pi_{susp}^{m}\): モード別固定姿勢

\(\pi_{crouch}\)は障害物幾何、アンカー位置、ワイヤ長を入力としない。固有受容状態、IMU、深度指令から姿勢安定化アクションを出す。そのため、ワイヤ計画器から機能的に独立した運動スキルとして説明できる。

### 8.2 しゃがみ残差制御

しゃがみ深度を \(d\in[0,0.9]\)、デフォルト姿勢を \(\mathbf q_0\)、目標しゃがみ姿勢を \(\mathbf q_c\)、actor残差を \(\mathbf a_c\) とする。

\[
\mathbf q_d
=\mathbf q_0+d(\mathbf q_c-\mathbf q_0)
+\mathbf S_a\tilde{\mathbf a}_c.
\tag{39}
\]

深度に応じてhip/ankle roll残差を縮小し、waist yaw残差を0にする。報酬は次を同時に最適化する。

- root高さと前傾pitch
- torso roll/yaw抑制
- 全身COMの支持領域内配置
- 膝直下への足裏配置
- 目標足幅
- 足前後位置と速度の抑制
- 両足接地
- 最終姿勢での長時間静止
- 左右対称な腕姿勢

### 8.3 ポリシー切替

歩行としゃがみのアクションを

\[
\mathbf a
=(1-\beta)\mathbf a_w+\beta\mathbf a_c
\tag{40}
\]

でblendする。\(\beta\)はsmoothstep

\[
\beta(u)=3u^2-2u^3
\tag{41}
\]

を使い、深度0の状態でactor handoverを完了してからしゃがみ始める。

固定姿勢への切替は、ワイヤ支持成立後に実測関節姿勢 \(\mathbf q_m\) から

\[
\mathbf q_d(t)
=(1-\gamma(t))\mathbf q_m
+\gamma(t)\mathbf q_{susp}^m
\tag{42}
\]

と補間し、関節指令変化率を制限する。実測姿勢から始めることで、接地中のPD追従誤差を空中へ持ち越して脚が一度過剰に開く現象を抑える。

---

## 9. モード差を一表で示す

| 項目 | Ascend | Descend | Gap |
|---|---|---|---|
| 開始支持面 | 地面 | 台上 | 手前台上 |
| アンカー役割 | 上方＋前方へ吊上げ | edge外へ支持しながら降下 | 向こう岸への振り子支点 |
| LIFT feed-forward | (mg/u_z) | (mg/u_z) | (mg u_z)（現行） |
| 吊上げ量 | 段差高＋clearance | 小preload | 衝突回避＋振り子支持 |
| 固定姿勢 | 開脚wide | 足を前下方 | 屈膝・開脚・左右対称 |
| HOLD終了 | 時間 | 時間 | 向こう岸位置＋整定 |
| LOWER目標 | 台上 | 下側床 | 向こう岸台上 |

この表と状態遷移図を組み合わせれば、3モードを別々のアルゴリズムとして説明せずに済む。

---

## 10. Supplementary用Algorithmテンプレート

次の擬似コードはSupplementaryまたは原稿作成時の説明骨格として使用する。6ページ版の本文では、同じ情報をFig. 2の状態遷移図と短い本文に置き換える方が紙面効率が高い。査読者からアルゴリズムの曖昧さを指摘される場合、結果図を圧迫しない範囲で15–20行のAlgorithmとして戻す。

```text
Algorithm 1: Hybrid wire-assisted traversal
Input: mode m, known geometry G, anchor p_a, safety margins μ_m
1:  σ ← APPROACH,  L_c ← measured cable length
2:  while σ ≠ DONE do
3:      acquire proprioception, IMU, L, T
4:      update attachment geometry (p_b, u, Ldot)
5:      select humanoid skill and velocity command from σ
6:      if σ = SETTLE and stability guard holds then
7:          (L_H, L_R) ← geometricPlanner(m, G, p_a, p_b)
8:          σ ← LIFT
9:      end if
10:     L_c, Ldot_c ← boundedLengthProfile(σ, L_H, L_R)
11:     if m = GAP and σ ∈ {LIFT,HOLD} then
12:         T_d ← swingTension(L,Ldot,L_c,Ldot_c)
13:     else
14:         T_d ← supportTension(L,Ldot,L_c,Ldot_c)
15:     end if
16:     T ← magnitudeAndSlewLimit(T_d)
17:     apply F_w=T u and τ_w=(p_b-p_com)×F_w
18:     update σ using length, rate, contact, and traversal guards
19:  end while
```

コードそのものはSupplementary Materialへ回す。論文本文で数十行のPython listingを使うと、数式・図・結果の面積を圧迫する。

---

## 11. 6ページの推奨配分

参考文献を含む6ページを想定した目安である。Humanoids 2022の公式投稿案内は「図および参考文献を含め6ページ」としていたため、対象年度の最終Call for Papersが公開されるまでは、この保守的な条件で組版する。

| 節 | 量 | 内容 |
|---|---:|---|
| Page 1: title, abstract, Introduction | 1.00 page | 背景、先行研究、gap/stepの限界、貢献3点 |
| Method | 1.60 | 力学、長さ・張力制御、状態機械、しゃがみskill |
| Experiments | 2.10 | setup 0.40、主要結果1.25、比較・ablation 0.45 |
| Discussion/Limitations | 0.45 | 既知幾何、理想ワイヤ、一般化、実機化 |
| Conclusion | 0.20 | 結果と意義を一段落で要約 |
| References | 0.65 | 約18–25件を想定 |

合計は6.00ページである。独立したRelated Work節は作らず、Introductionの中へ約0.20–0.25ページで統合する。Abstractは150–180語程度、Introduction本文は残り約0.70ページを使う。

### ページ単位の配置

```text
Page 1.0       Title + Abstract + Introduction (related workを含む)
Page 1.0–2.6   Method
Page 2.6–4.7   Experiments (setup + results + ablation)
Page 4.7–5.15  Discussion and Limitations
Page 5.15–5.35 Conclusion
Page 5.35–6.0  References
```

### Introduction 1ページの内訳

- 0.15 page: 問題背景と「単体歩行では越えられない地形」
- 0.20 page: 関連研究と不足点
- 0.20 page: 外部ワイヤ支持という着想
- 0.15 page: 本研究の問題設定と既知環境仮定
- 0.20 page: 貢献3点と論文構成

### Method 1.6ページの内訳

- 0.30 page: システムモデル、座標、式(1)、式(4)–(6)
- 0.40 page: 幾何長計画と長さ軌道、式(9)–(10)、式(21)
- 0.35 page: 通常・gap張力制御、式(27)、式(29)、式(33)
- 0.30 page: 共通状態機械
- 0.25 page: しゃがみskillとpolicy/fixed-posture切替

### Experiments 2.1ページの内訳

- 0.40 page: G1、地形、アンカー、制御周期、評価条件
- 0.55 page: ascend/descend/gapの定量結果表
- 0.45 page: 長さ・張力・姿勢の代表時系列図
- 0.45 page: 無補助、旧張力制御、しゃがみなし等とのablation
- 0.25 page: 成功・失敗例と短い解析

Method節では次を使う。

- Fig. 1: システム全体と座標・アンカー・背部取付点
- Fig. 2: 共通状態機械と各状態のpolicy/winch動作
- Table I: 3モードの環境・アンカー・長さ・速度・張力設定

Fig. 1とFig. 2は可能なら一つの二段figureへ統合する。Algorithm 1、固定姿勢の全関節角、全報酬重みは本文に入れず、Supplementaryまたはプロジェクトページへ置く。本文では重要な設計目標だけを書く。

本文へ残す数式も絞る。推奨するのは、ロボット運動方程式(1)、ワイヤ幾何とwrench(4)–(6)、共通目標長(9)–(10)、長さ速度則(21)、通常張力(27)、gap張力(29)、張力制約(33)である。残りの式は説明文、表、Supplementaryへ移す。これならMethodを約1.6ページに収められる。

---

## 12. Method節の短い英語ドラフト

### A. System Model

> We consider a floating-base humanoid equipped with a single massless tensile cable attached to a fixed point on the rear upper torso. The anchor position and terrain geometry are assumed to be known a priori. Let \(\mathbf p_a\) and \(\mathbf p_b\) denote the world-frame anchor and attachment positions, respectively. The cable length and direction are \(L=\|\mathbf p_a-\mathbf p_b\|\) and \(\hat{\mathbf u}=(\mathbf p_a-\mathbf p_b)/L\). A unilateral tension \(T\ge0\) produces \(\mathbf F_w=T\hat{\mathbf u}\). In simulation, the equivalent torso wrench includes the moment \(\boldsymbol\tau_w=(\mathbf p_b-\mathbf p_{com,t})\times\mathbf F_w\), thereby preserving the rotational effect of the rear attachment.

### B. Geometry-Based Cable Planning

> For each traversal mode, the known terrain geometry defines a hoisting waypoint \(\mathbf p_{b,H}^{m}\) and a supported-landing waypoint \(\mathbf p_{b,R}^{m}\) for the attachment point. The corresponding cable targets are \(L_H^m=\|\mathbf p_a-\mathbf p_{b,H}^{m}\|+\delta_H^m\) and \(L_R^m=\max(\|\mathbf p_a-\mathbf p_{b,R}^{m}\|+\delta_R^m,L_H^m)\), where \(\delta_H^m\) and \(\delta_R^m\) account for calibrated routing and model uncertainty. Hence, reel-in and payout distances are planner outputs rather than manually specified primary inputs.

### C. Length and Tension Control

> The winch command follows a bounded-speed stopping profile, \(\dot L_c=\operatorname{sgn}(L_g-L_c)\min(v_{max},\sqrt{2a_b|L_g-L_c|})\). For quasi-static hoisting and lowering, the desired tension is \(T_d=[m_tg/u_z+K_p(L-L_c)+K_d(\dot L_f-\dot L_c)]_+\), where \(u_z\) is the vertical component of the cable direction. During gap traversal, the feedforward term is changed to the radial pendulum component \(m_tgu_z\), allowing gravity to accelerate the robot tangentially toward the far platform. Tension magnitude and slew rate are saturated to represent winch limits and to avoid impulsive load transfer.

### D. Hybrid Skill Coordination

> A finite-state supervisor coordinates APPROACH, SETTLE, LIFT, HOLD, LOWER, ASSIST, RELEASE, and DONE phases. A pretrained locomotion policy approaches the edge and recovers after landing. Before hoisting, a residual crouching policy moves the robot to a planted, forward-leaning, wide-stance posture while penalizing sagittal foot displacement, base drift, torso roll/yaw, and loss of double support. Once cable support is established, the controller transitions from the measured joint configuration to a mode-specific symmetric suspension posture. During ASSIST, the landing cable length is held while the standing policy is blended back in; the cable is released only after this recovery transition.

---

## 13. 本文で報告すべき制御パラメータ

全Python引数を掲載する必要はない。最低限、次をTable IまたはSupplementaryに載せる。

- アンカー \((x_a,y_a,z_a)\)
- 背部取付点 \(\mathbf r_b^B\)
- 地形高、gap幅
- $L_0,L_H,L_R$ またはその平均・代表値
- $v_{reel,max},v_{pay,max},a_{b,reel},a_{b,pay}$
- $K_p,K_d,K_{p,g},K_{d,g}$
- $T_{max},\dot T_{max},\dot T_{release}$
- length/rate整定閾値
- しゃがみ深度・遷移時間・安定判定時間
- policy復帰blend時間
- モード別固定姿勢はSupplementaryに全角度を掲載

現行代表値:

| Parameter | Value |
|---|---:|
| attachment offset | $(-0.14,0,0.40)$ m |
| ordinary $K_p,K_d$ | 500 N/m, 350 N·s/m |
| gap $K_{p,g},K_{d,g}$ | 5000 N/m, 1000 N·s/m |
| ordinary max tension | 470.7 N |
| ascend max tension | 392.3 N |
| ordinary tension slew | 1200 N/s |
| release slew | 200 N/s |
| length tolerance | 0.025 m |
| radial-rate tolerance | 0.06 m/s |
| crouch depth | 0.90 |
| crouch transition | 4.0 s |
| crouch stability window | 0.75 s |
| policy recovery blend | 0.8 s |

---

## 14. 添付コードの適切な量

6ページ本文にはコードlistingを入れない。状態遷移は統合figureと本文で説明し、擬似コードと参照実装は添付・Supplementaryへ置く。Supplementaryには次の二つを用意する。

1. `doc/paper_wire_controller_reference.py`
   - 約200行
   - 依存はNumPyのみ
   - 幾何計画、長さ軌道、張力式、slew limit、wrenchを収録
   - 論文数式との対応が明確
2. 実験再現コードへのリンク
   - `scripts/play.py`
   - `src/tasks/crouch/`
   - checkpoint、commit hash、実行コマンド

参照コードは読みやすさを優先した数式対応版であり、production実装の全例外処理、MuJoCo indexing、viewer、logger、policy loaderを含めない。これらは科学的な制御則の理解には不要で、紙面と可読性を損なう。

---

## 15. 現行コードとの整合性チェック

論文提出前に、次を必ず解決する。

1. descend/gapの目標長を式(9)–(10)の3次元距離へ合わせるか、近似式を本文に記載する。
2. gapの経験的offsetをcalibration marginとして明示し、感度を示す。
3. gap式(28)へ向心力項を追加するか、現行どおり省略した式を採用する。
4. 最終成功checkpointを一つに固定する。
5. `no_terminations=True`と独立した成功判定を実装する。
6. 実験commitをcleanな状態でtag付けする。
7. CADを含めない場合、質量なし理想ワイヤ・理想取付点の原理検証であると明記する。

学術的にきれいな式へ整えることと、実験に使った実装を正確に報告することは別問題である。論文の制御式は、最終的に走らせたコードと一致させる必要がある。
