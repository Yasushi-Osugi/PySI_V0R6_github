#main250114.py


#   PySI_V0R6_github/
#   ├── main.py                # エントリーポイント（GUIの起動）
#   ├── README.md
#   ├── .gitignore
#   └── src/
#       ├── submodule1/

#├── gui/
#│   ├── app.py             # GUIロジックと画面レイアウト
#│   └── psi_graph.py       # PSIグラフの描画
#├── network/
#│   ├── tree.py            # サプライチェーンのツリー構造操作
#│   └── network_graph.py   # NetworkXグラフの生成と操作
#├── utils/
#│   ├── file_io.py         # ファイルの入出力ユーティリティ
#│   ├── demand_processing.py # 需要データの処理
#│   └── config.py          # 設定情報と定数


#project/
#├── main.py                # エントリーポイント（GUIの起動）
#├── gui/
#│   ├── app.py             # GUIロジックと画面レイアウト
#│   └── psi_graph.py       # PSIグラフの描画
#├── network/
#│   ├── tree.py            # サプライチェーンのツリー構造操作
#│   └── network_graph.py   # NetworkXグラフの生成と操作
#├── utils/
#│   ├── file_io.py         # ファイルの入出力ユーティリティ
#│   ├── demand_processing.py # 需要データの処理
#│   └── config.py          # 設定情報と定数
#└── tests/
#    ├── test_tree.py       # ツリー構造のテスト
#    ├── test_demand.py     # 需要データ処理のテスト
#    └── test_gui.py        # GUI操作のテスト
#モジュール内容の詳細
#main.py
#
#プロジェクトのエントリーポイント。
#PSIPlannerApp（GUIアプリ）の起動。
#gui/app.py
#
#setup_uiや各種イベントハンドラを含む。
#ユーザーからの操作（ファイル選択やボタンのクリック）に応じて他モジュールを呼び#出す。
#gui/psi_graph.py
#
#PSIグラフ（右側の描画エリア）を描画するロジックを定義。
#show_psiメソッドをモジュールとして独立。
#network/tree.py
#
#サプライチェーンのツリー構造の作成・操作（ノード追加、属性設定など）。
#例: create_tree_set_attribute。
#network/network_graph.py
#
#NetworkXを使ったネットワークグラフ（左側の描画エリア）の生成・描画ロジック。
#例: view_nx_matlib4opt。
#utils/file_io.py
#
#ファイルの入出力関連のユーティリティ関数。
#例: load_csv, save_csv。
#utils/demand_processing.py
#
#需要データの処理ロジック（例: generate_weekly_demand）。
#ロットIDの生成や週次データへの変換。
#utils/config.py
#
#定数（LOT_SIZE や PLAN_YEARS）や設定情報。
#tests ディレクトリ
#
#各モジュールのテストコードを格納。
#pytestを使ってユニットテストを実行可能。
#




# main.py
from pysi.utils.config import Config

from pysi.gui.app import *


import tkinter as tk

def main():
    # Create a global configuration instance
    config = Config()

    # Initialize GUI with the configuration
    root = tk.Tk()
    app = PSIPlannerApp(root, config)
    root.mainloop()

if __name__ == "__main__":
    main()


