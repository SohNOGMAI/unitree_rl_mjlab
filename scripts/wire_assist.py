import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

class WireAssistController:
    def __init__(self, model, data, body_name="torso_link"):
        """
        ワイヤーアシストの初期化
        """
        self.model = model
        self.data = data
        
        # 1. 力を加える対象（胴体）のIDを取得する
        # MuJoCoではパーツを「名前」ではなく「ID（数字）」で管理しているため、それを検索します。
        self.body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id == -1:
            print(f"⚠️ 警告: '{body_name}' という名前のボディが見つかりません！")
        else:
            print(f"✅ ワイヤー接続完了: 胴体IDは {self.body_id} です。")

        # 2. ワイヤーの「アシスト設定」を定義
        self.pitch_threshold = 0.3  # 介入を開始する傾き（ラジアン。0.3rad ≒ 約17度）
        self.pull_force_z = 80.0    # 上に引っ張る力（ニュートン）
        self.pull_force_x = -30.0   # 後ろに引っ張る力（ニュートン）

    def step(self):
        """
        シミュレーションの1ステップごとに呼ばれる関数
        """
        if self.body_id == -1:
            return

        # --- A. 現在の姿勢を読み取る ---
        # 胴体のクォータニオン（姿勢データ）を取得し、計算しやすいオイラー角（ピッチ・ロール・ヨー）に変換します。
        quat = self.data.xquat[self.body_id]
        # scipyの仕様に合わせて [x, y, z, w] の順序に並び替え
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]]) 
        euler = r.as_euler('xyz', degrees=False)
        pitch = euler[1] # Y軸周りの回転（前後の傾き）

        # --- B. 加える力の配列を準備する ---
        # [X軸方向の力, Y軸の力, Z軸の力, X軸のトルク, Y軸のトルク, Z軸のトルク] の6次元配列
        applied_force = np.zeros(6)

        # --- C. 閾値を超えたら「見えない糸」で引っ張る ---
        if pitch > self.pitch_threshold:
            # 前に倒れそうになっている場合 -> 「上」と「後ろ」に引っ張る
            applied_force[0] = self.pull_force_x 
            applied_force[2] = self.pull_force_z
            print(f"⚠️ 前方転倒の危機！ (Pitch: {pitch:.2f}) -> ワイヤーで引き戻します！")
            
        elif pitch < -self.pitch_threshold:
            # 後ろに倒れそうになっている場合 -> 「上」と「前」に引っ張る
            applied_force[0] = -self.pull_force_x 
            applied_force[2] = self.pull_force_z
            print(f"⚠️ 後方転倒の危機！ (Pitch: {pitch:.2f}) -> ワイヤーで引き戻します！")

        # --- D. MuJoCoの物理エンジンに力を送信 ---
        # 毎ステップ、この配列の力が指定したボディ（胴体）に加わります。
        # （閾値を超えていない時は applied_force は全て0なので、何も力は加わりません）
        self.data.xfrc_applied[self.body_id] = applied_force