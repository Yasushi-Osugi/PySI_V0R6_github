#250114gui_app.py

# gui/app.py

# ********************************
# library import
# ********************************
import os
import shutil
import threading

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont, Tk, Menu, ttk

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


import networkx as nx


import datetime as dt
from datetime import datetime as dt_datetime, timedelta

import math

import copy
import pickle

# ********************************
# library import
# ********************************

import mpld3
from mpld3 import plugins

from collections import defaultdict


import numpy as np

from dateutil.relativedelta import relativedelta

import calendar




# ********************************
# PySI library import
# ********************************
from pysi.utils.config import Config

from pysi.utils.file_io import *
#from utils.file_io import load_cost_table
#from utils.file_io import load_monthly_demand

from pysi.plan.demand_generate import convert_monthly_to_weekly

from pysi.plan.operations import *
# "plan.demand_processing" is merged in "plan.operations"

#from plan.demand_processing import *
#from pysi.plan.demand_processing import set_df_Slots2psi4demand


from pysi.network.tree import *
#from network.tree import create_tree_set_attribute
#from network.tree import set_node_costs

#from network.tree import calc_all_psi2i4demand, set_lot_counts

#from PSI_plan.planning_operation import calcS2P, set_S2psi, get_set_childrenP2S2psi, calc_all_psi2i4demand, calcPS2I4demand




# ********************************
# Definition start
# ********************************






def find_all_paths(node, path, paths):
    path.append(node.name)

    if not node.children:

        #print("leaf path", node.name, path)
        paths.append(path.copy())

    else:

        for child in node.children:

            # print("child path",child.name, path)

            find_all_paths(child, path, paths)

            for grandchild in child.children:
                #print("grandchild path", grandchild.name, path)
                find_all_paths(grandchild, path, paths)

                for g_grandchild in grandchild.children:
                    #print("g_grandchild path", g_grandchild.name, path)
                    find_all_paths(g_grandchild, path, paths)

                    for g1_grandchild in g_grandchild.children:
                        #print("g1_grandchild path", g1_grandchild.name, path)
                        find_all_paths(g1_grandchild, path, paths)

                        for g2_grandchild in g1_grandchild.children:
                            #print("g2_grandchild path", g2_grandchild.name, path)
                            find_all_paths(g2_grandchild, path, paths)

    path.pop()


def find_paths(root):

    paths = []

    find_all_paths(root, [], paths)

    return paths





# *********************************
# check_plan_range
# *********************************
def check_plan_range(df):  # df is dataframe

    #
    # getting start_year and end_year
    #
    start_year = node_data_min = df["year"].min()
    end_year = node_data_max = df["year"].max()

    # *********************************
    # plan initial setting
    # *********************************

    plan_year_st = int(start_year)  # 2024  # plan開始年

    # 3ヵ年または5ヵ年計画分のS計画を想定
    plan_range = int(end_year) - int(start_year) + 1 + 1  # +1はハミ出す期間

    plan_year_end = plan_year_st + plan_range

    return plan_range, plan_year_st


# 2. lot_id_list列を追加
def generate_lot_ids(row):

    # node_yyyy_ww = f"{row['node_name']}_{row['iso_year']}_{row['iso_week']}"
    node_yyyy_ww = f"{row['node_name']}{row['iso_year']}{row['iso_week']}"

    lots_count = row["S_lot"]

    # stack_list = [f"{node_yyyy_ww}_{i}" for i in range(lots_count)]

    #@240930 修正MEMO
    # ココの{i}をzfillで二桁にする
    #stack_list = [f"{node_yyyy_ww}{i:02}" for i in range(lots_count)]

    digit_count = 2
    stack_list = [f"{node_yyyy_ww}{str(i).zfill(digit_count)}" for i in range(lots_count)]

    return stack_list




# ******************************
# trans month 2 week 2 lot_id_list
# ******************************
def trans_month2week2lot_id_list(file_name, lot_size):

    df = pd.read_csv(file_name)

    # *********************************
    # check_plan_range
    # *********************************
    plan_range, plan_year_st = check_plan_range(df)  # df is dataframe

    df = df.melt(
        id_vars=["product_name", "node_name", "year"],
        var_name="month",
        value_name="value",
    )

    df["month"] = df["month"].str[1:].astype(int)

    df_daily = pd.DataFrame()

    for _, row in df.iterrows():

        daily_values = np.full(
            pd.Timestamp(row["year"], row["month"], 1).days_in_month, row["value"]
        )

        dates = pd.date_range(
            start=f"{row['year']}-{row['month']}-01", periods=len(daily_values)
        )

        df_temp = pd.DataFrame(
            {
                "product_name": row["product_name"],
                "node_name": row["node_name"],
                "date": dates,
                "value": daily_values,
            }
        )

        df_daily = pd.concat([df_daily, df_temp])


    #@24240930 STOP
    #df_daily["iso_year"] = df_daily["date"].dt.isocalendar().year
    #df_daily["iso_week"] = df_daily["date"].dt.isocalendar().week
    #
    #df_weekly = (
    #    df_daily.groupby(["product_name", "node_name", "iso_year", "iso_week"])["value"]
    #    .sum()
    #    .reset_index()
    #)


    df_daily["iso_year"] = df_daily["date"].dt.isocalendar().year

    # ISO週を２ケタ表示
    df_daily["iso_week"] = df_daily["date"].dt.isocalendar().week.astype(str).str.zfill(2)

    df_weekly = (
        df_daily.groupby(["product_name", "node_name", "iso_year", "iso_week"])["value"]
        .sum()
        .reset_index()
    )

    ## 1. S_lot列を追加
    # lot_size = 100  # ここに適切なlot_sizeを設定します
    df_weekly["S_lot"] = df_weekly["value"].apply(lambda x: math.ceil(x / lot_size))

    ## 2. lot_id_list列を追加
    # def generate_lot_ids(row):
    df_weekly["lot_id_list"] = df_weekly.apply(generate_lot_ids, axis=1)

    return df_weekly, plan_range, plan_year_st




def make_capa_year_month(input_file):

    #    # mother plant capacity parameter
    #    demand_supply_ratio = 1.2  # demand_supply_ratio = ttl_supply / ttl_demand

    # initial setting of total demand and supply
    # total_demandは、各行のm1からm12までの列の合計値

    df_capa = pd.read_csv(input_file)

    df_capa["total_demand"] = df_capa.iloc[:, 3:].sum(axis=1)

    # yearでグループ化して、月次需要数の総和を計算
    df_capa_year = df_capa.groupby(["year"], as_index=False).sum()

    return df_capa_year




#def trans_month2week2lot_id_list(file_name, lot_size)
def process_monthly_demand(file_name, lot_size):
    """
    Process monthly demand data and convert to weekly data.

    Parameters:
        file_name (str): Path to the monthly demand file.
        lot_size (int): Lot size for allocation.

    Returns:
        pd.DataFrame: Weekly demand data with ISO weeks and lot IDs.
    """
    monthly_data = load_monthly_demand(file_name)
    if monthly_data.empty:
        print("Error: Failed to load monthly demand data.")
        return None

    return convert_monthly_to_weekly(monthly_data, lot_size)




def read_set_cost(file_path, nodes_outbound):
    """
    Load cost table from file and set node costs.

    Parameters:
        file_path (str): Path to the cost table file.
        nodes_outbound (dict): Dictionary of outbound nodes.

    Returns:
        None
    """
    cost_table = load_cost_table(file_path)
    if cost_table is not None:
        set_node_costs(cost_table, nodes_outbound)




# ****************************
# 辞書をtree nodeのdemand & supplyに接続する
# ****************************
def set_dict2tree_psi(node, attr_name, node_psi_dict):

    setattr(node, attr_name, node_psi_dict.get(node.name))

    # node.psi4supply = node_psi_dict.get(node.name)

    for child in node.children:

        set_dict2tree_psi(child, attr_name, node_psi_dict)


# nodeを手繰りながらnode_psi_dict辞書を初期化する
def make_psi_space_dict(node, node_psi_dict, plan_range):

    psi_list = [[[] for j in range(4)] for w in range(53 * plan_range)]

    node_psi_dict[node.name] = psi_list  # 新しいdictにpsiをセット

    for child in node.children:

        make_psi_space_dict(child, node_psi_dict, plan_range)

    return node_psi_dict




# *******************
# 生産平準化の前処理　ロット・カウント
# *******************
def count_lots_yyyy(psi_list, yyyy_str):

    matrix = psi_list

    # 共通の文字列をカウントするための変数を初期化
    count_common_string = 0

    # Step 1: マトリクス内の各要素の文字列をループで調べる
    for row in matrix:

        for element in row:

            # Step 2: 各要素内の文字列が "2023" を含むかどうかを判定
            if yyyy_str in element:

                # Step 3: 含む場合はカウンターを増やす
                count_common_string += 1

    return count_common_string


def is_52_or_53_week_year(year):
    # 指定された年の12月31日を取得
    last_day_of_year = dt.date(year, 12, 31)

    # 12月31日のISO週番号を取得 (isocalendar()メソッドはタプルで[ISO年, ISO週番号, ISO曜日]を返す)
    _, iso_week, _ = last_day_of_year.isocalendar()

    # ISO週番号が1の場合は前年の最後の週なので、52週と判定
    if iso_week == 1:
        return 52
    else:
        return iso_week


def find_depth(node):
    if not node.parent:
        return 0
    else:
        return find_depth(node.parent) + 1


def find_all_leaves(node, leaves, depth=0):
    if not node.children:
        leaves.append((node, depth))  # (leafノード, 深さ) のタプルを追加
    else:
        for child in node.children:
            find_all_leaves(child, leaves, depth + 1)


def make_nodes_decouple_all(node):

    #
    #    root_node = build_tree()
    #    set_parent(root_node)

    #    leaves = []
    #    find_all_leaves(root_node, leaves)
    #    pickup_list = leaves[::-1]  # 階層の深い順に並べる

    leaves = []
    leaves_name = []

    nodes_decouple = []

    find_all_leaves(node, leaves)
    # find_all_leaves(root_node, leaves)
    pickup_list = sorted(leaves, key=lambda x: x[1], reverse=True)
    pickup_list = [leaf[0] for leaf in pickup_list]  # 深さ情報を取り除く

    # こうすることで、leaf nodeを階層の深い順に並べ替えた pickup_list が得られます。
    # 先に深さ情報を含めて並べ替え、最後に深さ情報を取り除くという流れになります。

    # 初期処理として、pickup_listをnodes_decoupleにcopy
    # pickup_listは使いまわしで、pop / insert or append / removeを繰り返す
    for nd in pickup_list:
        nodes_decouple.append(nd.name)

    nodes_decouple_all = []

    while len(pickup_list) > 0:

        # listのcopyを要素として追加
        nodes_decouple_all.append(nodes_decouple.copy())

        current_node = pickup_list.pop(0)
        del nodes_decouple[0]  # 並走するnode.nameの処理

        parent_node = current_node.parent

        if parent_node is None:
            break

        # 親ノードをpick up対象としてpickup_listに追加
        if current_node.parent:

            #    pickup_list.append(current_node.parent)
            #    nodes_decouple.append(current_node.parent.name)

            # if parent_node not in pickup_list:  # 重複追加を防ぐ

            # 親ノードの深さを見て、ソート順にpickup_listに追加
            depth = find_depth(parent_node)
            inserted = False

            for idx, node in enumerate(pickup_list):

                if find_depth(node) <= depth:

                    pickup_list.insert(idx, parent_node)
                    nodes_decouple.insert(idx, parent_node.name)

                    inserted = True
                    break

            if not inserted:
                pickup_list.append(parent_node)
                nodes_decouple.append(parent_node.name)

            # 親ノードから見た子ノードをpickup_listから削除
            for child in parent_node.children:

                if child in pickup_list:

                    pickup_list.remove(child)
                    nodes_decouple.remove(child.name)

        else:

            print("error: node dupplicated", parent_node.name)

    return nodes_decouple_all






    # +++++++++++++++++++++++++++++++++++++++++++++++
    # Mother Plant demand leveling 
    # +++++++++++++++++++++++++++++++++++++++++++++++
def demand_leveling_on_ship(root_node_outbound, pre_prod_week, year_st, year_end):

    # input: root_node_outbound.psi4demand
    #        pre_prod_week =26 
    #
    # output:root_node_outbound.psi4supply

    plan_range = root_node_outbound.plan_range


    #@241114
    # 需給バランスの問題は、ひとつ上のネットワーク全体のoptimizeで解く

    # ロット単位で供給を変化させて、weight=ロット(CPU_profit)利益でsimulate
    # 設備投資の回収期間を見る

    # 供給>=需要ならオペレーション問題
    # 供給<需要なら供給配分とオペレーション問題

    # optimiseで、ルートと量を決定
    # PSIで、operation revenue cost profitを算定 business 評価

    # 業界No1/2/3の供給戦略をsimulateして、business評価する


    # node_psi_dict_Ot4Dmでは、末端市場のleafnodeのみセット
    #
    # root_nodeのS psi_list[w][0]に、levelingされた確定出荷S_confirm_listをセッ    ト

    # 年間の総需要(総lots)をN週先行で生産する。
    # 例えば、３ヶ月先行は13週先行生産として、年間総需要を週平均にする。

    # S出荷で平準化して、confirmedS-I-P
    # conf_Sからconf_Pを生成して、conf_P-S-I  PUSH and PULL

    S_list = []
    S_allocated = []

    year_lots_list = []
    year_week_list = []

    leveling_S_in = []

    leveling_S_in = root_node_outbound.psi4demand

    # psi_listからS_listを生成する
    for psi in leveling_S_in:

        S_list.append(psi[0])

    # 開始年を取得する
    plan_year_st = year_st  # 開始年のセット in main()要修正

    for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):

        year_lots = count_lots_yyyy(S_list, str(yyyy))

        year_lots_list.append(year_lots)

    #        # 結果を出力
    #       #print(yyyy, " year carrying lots:", year_lots)
    #
    #    # 結果を出力
    #   #print(" year_lots_list:", year_lots_list)

    # an image of sample data
    #
    # 2023  year carrying lots: 0
    # 2024  year carrying lots: 2919
    # 2025  year carrying lots: 2914
    # 2026  year carrying lots: 2986
    # 2027  year carrying lots: 2942
    # 2028  year carrying lots: 2913
    # 2029  year carrying lots: 0
    #
    # year_lots_list: [0, 2919, 2914, 2986, 2942, 2913, 0]

    year_list = []

    for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):

        year_list.append(yyyy)

        # テスト用の年を指定
        year_to_check = yyyy

        # 指定された年のISO週数を取得
        week_count = is_52_or_53_week_year(year_to_check)

        year_week_list.append(week_count)

    #        # 結果を出力
    #       #print(year_to_check, " year has week_count:", week_count)
    #
    #    # 結果を出力
    #   #print(" year_week_list:", year_week_list)

    # print("year_list", year_list)

    # an image of sample data
    #
    # 2023  year has week_count: 52
    # 2024  year has week_count: 52
    # 2025  year has week_count: 52
    # 2026  year has week_count: 53
    # 2027  year has week_count: 52
    # 2028  year has week_count: 52
    # 2029  year has week_count: 52
    # year_week_list: [52, 52, 52, 53, 52, 52, 52]


    # *****************************
    # 生産平準化のための年間の週平均生産量(ロット数単位)
    # *****************************

    # *****************************
    # make_year_average_lots
    # *****************************
    # year_list     = [2023,2024,2025,2026,2027,2028,2029]

    # year_lots_list = [0, 2919, 2914, 2986, 2942, 2913, 0]
    # year_week_list = [52, 52, 52, 53, 52, 52, 52]

    year_average_lots_list = []

    for lots, weeks in zip(year_lots_list, year_week_list):

        average_lots_per_week = math.ceil(lots / weeks)

        year_average_lots_list.append(average_lots_per_week)


    # print("year_average_lots_list", year_average_lots_list)
    #
    # an image of sample data
    #
    # year_average_lots_list [0, 57, 57, 57, 57, 57, 0]

    # 年間の総需要(総lots)をN週先行で生産する。
    # 例えば、３ヶ月先行は13週先行生産として、年間総需要を週平均にする。

    #
    # 入力データの前提
    #
    # leveling_S_in[w][0] == S_listは、outboundのdemand_planで、
    # マザープラントの出荷ポジションのSで、
    # 5年分 週次 最終市場におけるlot_idリストが
    # LT offsetされた状態で入っている
    #
    # year_list     = [2023,2024,2025,2026,2027,2028,2029]

    # year_lots_list = [0, 2919, 2914, 2986, 2942, 2913, 0]
    # year_week_list = [52, 52, 52, 53, 52, 52, 52]
    # year_average_lots_list [0, 57, 57, 57, 57, 57, 0]

    # ********************************
    # 先行生産の週数
    # ********************************
    # precedence_production_week =13

    pre_prod_week =26 # 26週=6か月の先行生産をセット
    # pre_prod_week =13 # 13週=3か月の先行生産をセット
    # pre_prod_week = 6  # 6週=1.5か月の先行生産をセット

    # ********************************
    # 先行生産の開始週を求める
    # ********************************
    # 市場投入の前年において i= 0  year_list[i]           # 2023
    # 市場投入の前年のISO週の数 year_week_list[i]         # 52

    # 先行生産の開始週は、市場投入の前年のISO週の数 - 先行生産週

    pre_prod_start_week = 0

    i = 0

    pre_prod_start_week = year_week_list[i] - pre_prod_week

    # スタート週の前週まで、[]リストで埋めておく
    for i in range(pre_prod_start_week):
        S_allocated.append([])

    # ********************************
    # 最終市場からのLT offsetされた出荷要求lot_idリストを
    # Allocate demand to mother plant weekly slots
    # ********************************

    # S_listの週別lot_idリストを一直線のlot_idリストに変換する
    # mother plant weekly slots

    # 空リストを無視して、一直線のlot_idリストに変換

    # 空リストを除外して一つのリストに結合する処理
    S_one_list = [item for sublist in S_list if sublist for item in sublist]

    ## 結果表示
    ##print(S_one_list)

    # to be defined 毎年の定数でのlot_idの切り出し

    # listBの各要素で指定された数だけlistAから要素を切り出して
    # 新しいリストlistCを作成

    listA = S_one_list  # 5年分のlot_idリスト

    listB = year_lots_list  # 毎年毎の総ロット数

    listC = []  # 毎年のlot_idリスト

    start_idx = 0

    for i, num in enumerate(listB):

        end_idx = start_idx + num

        # original sample
        # listC.append(listA[start_idx:end_idx])

        # **********************************
        # "slice" and "allocate" at once
        # **********************************
        sliced_lots = listA[start_idx:end_idx]

        # 毎週の生産枠は、year_average_lots_listの平均値を取得する。
        N = year_average_lots_list[i]

        if N == 0:

            pass

        else:

            # その年の週次の出荷予定数が生成される。
            S_alloc_a_year = [
                sliced_lots[j : j + N] for j in range(0, len(sliced_lots), N)
            ]

            S_allocated.extend(S_alloc_a_year)
            # S_allocated.append(S_alloc_a_year)

        start_idx = end_idx

    ## 結果表示
    # print("S_allocated", S_allocated)

    # set psi on outbound supply

    # "JPN-OUT"
    #


    # ***********************************************
    #@241113 CHANGE root_node_outbound.psi4supplyが存在するという前提
    # ***********************************************
    #
    #node_name = root_node_outbound.name  # Nodeからnode_nameを取出す
    #
    ## for w, pSi in enumerate( S_allocated ):
    ##
    ##    node_psi_dict_Ot4Sp[node_name][w][0] = pSi
    
    for w in range(53 * plan_range):

        if w <= len(S_allocated) - 1:  # index=0 start

            root_node_outbound.psi4supply[w][0] = S_allocated[w]
            #node_psi_dict_Ot4Sp[node_name][w][0] = S_allocated[w]

        else:

            root_node_outbound.psi4supply[w][0] = []
            #node_psi_dict_Ot4Sp[node_name][w][0] = []

    # +++++++++++++++++++++++++++++++++++++++++++++++






def place_P_in_supply_LT(w, child, lot):  # lot LT_shift on P

    # *******************************************
    # supply_plan上で、PfixをSfixにPISでLT offsetする
    # *******************************************

    # **************************
    # Safety Stock as LT shift
    # **************************

    #@240925 STOP
    ## leadtimeとsafety_stock_weekは、ここでは同じ
    ## safety_stock_week = child.leadtime
    #LT_SS_week = child.leadtime


    #@240925 長期休暇がLT_SS_weekかchild.leadtimeかどちらにある場合は???

    #@240925
    # leadtimeとsafety_stock_weekは別もの
    LT_SS_week   = child.safety_stock_week
    LT_logi_week = child.leadtime



    # **************************
    # long vacation weeks
    # **************************
    lv_week = child.long_vacation_weeks

    ## P to S の計算処理
    # self.psi4supply = shiftP2S_LV(self.psi4supply, safety_stock_week, lv_week)

    ### S to P の計算処理
    ##self.psi4demand = shiftS2P_LV(self.psi4demand, safety_stock_week, lv_week)

    # my_list = [1, 2, 3, 4, 5]
    # for i in range(2, len(my_list)):
    #    my_list[i] = my_list[i-1] + my_list[i-2]

    # 0:S
    # 1:CO
    # 2:I
    # 3:P


    #@240925 STOP
    ## LT:leadtime SS:safty stockは1つ
    ## foreward planで、「親confirmed_S出荷=子confirmed_P着荷」と表現
    #eta_plan = w + LT_SS_week  # ETA=ETDなので、+LTすると次のETAとなる


    # LT_logi_weekで子nodeまでの物流LTを考慮
    eta_plan = w + LT_logi_week


    # etd_plan = w + ss # ss:safty stock
    # eta_plan = w - ss # ss:safty stock

    # *********************
    # 着荷週が事業所nodeの非稼働週の場合 +1次週の着荷とする
    # *********************
    # 着荷週を調整
    eta_shift = check_lv_week_fw(lv_week, eta_plan)  # ETA:Eatimate Time Arriv

    # リスト追加 extend
    # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

    # lot by lot operation
    # confirmed_P made by shifting parent_conf_S

    # ***********************
    # place_lot_supply_plan
    # ***********************

    # ここは、"REPLACE lot"するので、appendの前にchild psiをzero clearしてから

    #@240925 STOP
    ## 今回のmodelでは、輸送工程もpsi nodeと同等に扱っている(=POではない)ので
    ## 親のconfSを「そのままのWで」子のconfPに置く place_lotする
    #child.psi4supply[w][3].append(lot)

    ## 親のconfSを「eta_shiftしたWで」子のconfPに置く place_lotする
    # 親のconfSを「LT=輸送LT + 加工LT + SSでwをshiftして」子confSにplace_lot

    child.psi4supply[eta_shift][3].append(lot)

    # print("len(child.psi4supply)", len(child.psi4supply) ) # len() of psi list    # print("lot child.name eta_shift ",lot,  child.name, eta_shift )  # LT shift weeks


    # Sは、SS在庫分の後に出荷する
    ship_position = eta_shift + LT_SS_week

    # 出荷週を調整
    ship_shift = check_lv_week_fw(lv_week, ship_position) 

    child.psi4supply[ship_shift][0].append(lot)




def find_path_to_leaf_with_parent(node, leaf_node, current_path=[]):

    current_path.append(leaf_node.name)

    if node.name == leaf_node.name:

        return current_path

    else:

        parent = leaf_node.parent

        path = find_path_to_leaf_with_parent(node, parent, current_path.copy())

    return path


#        if path:
#
#            return path


def extract_node_name(stringA):
    """
    Extract the node name from a string by removing the last 9 characters (YYYYWWNNN).

    Parameters:
        stringA (str): Input string (e.g., "LEAF01202601001").

    Returns:
        str: Node name (e.g., "LEAF01").
    """
    if len(stringA) > 9:
        # 最後の9文字を削除して返す # deep relation on "def generate_lot_ids()"
        return stringA[:-9]
    else:
        # 文字列が9文字以下の場合、削除せずそのまま返す（安全策）
        return stringA


#lot_IDが3桁の場合{i+1:03d}に、 9文字を削る
#lot_IDが4桁の場合{i+1:04d}に、10文字を削る
#10文字は、lot_IDが3桁の場合{i+1:03d}
#
# following "def generate_lot_ids()" is in "demand_generate.py"
#def generate_lot_ids(row):
#    """
#    Generate lot IDs based on the row's data.
#
#    Parameters:
#        row (pd.Series): A row from the DataFrame.
#
#    Returns:
#        list: List of generated lot IDs.
#    """
#    lot_count = row["S_lot"]
#
#    return [f"{row['node_name']}{row['iso_year']}{row['iso_week']}{i+1:03d}" for i in range(lot_count)]
#



def extract_node_name_OLD(stringA):
    # 右側の数字部分を除外してnode名を取得

    index = len(stringA) - 1

    while index >= 0 and stringA[index].isdigit():

        index -= 1

    node_name = stringA[: index + 1]

    return node_name





# ******************************************
# confirmedSを出荷先ship2のPとSにshift&set
# ******************************************
def feedback_psi_lists(node, nodes):
#def feedback_psi_lists(node, node_psi_dict, nodes):

    # キーが存在する場合は対応する値valueが返り、存在しない場合はNoneが返る。

    if node.children == []:  # 子nodeがないleaf nodeの場合

        pass

    else:

        # ************************************
        # clearing children P[w][3] and S[w][0]
        # ************************************
        # replace lotするために、事前に、
        # 出荷先となるすべてのchildren nodesのS[w][0]とP[w][3]をクリア

        for child in node.children:

            for w in range(53 * node.plan_range):

                child.psi4supply[w][0] = []
                child.psi4supply[w][3] = []

        # lotidから、leaf_nodeを特定し、出荷先ship2nodeに出荷することは、
        # すべての子nodeに出荷することになる

        # ************************************
        # setting mother_confirmed_S
        # ************************************
        # このnode内での子nodeへの展開
        for w in range(53 * node.plan_range):

            #@STOP
            #print("node.psi4supply", node.psi4supply)

            confirmed_S_lots = node.psi4supply[w][0]  # 親の確定出荷confS lot

            print("confirmed_S_lots", confirmed_S_lots)

            # 出荷先nodeを特定して

            # 一般には、下記のLT shiftだが・・・・・
            # 出荷先の ETA = LT_shift(ETD) でP place_lot
            # 工程中の ETA = SS_shift(ETD) でS place_lot

            # 本モデルでは、輸送工程 = modal_nodeを想定して・・・・・
            # 出荷先の ETA = 出荷元ETD        でP place_lot
            # 工程中の ETA = LT&SS_shift(ETD) でS place_lot
            # というイビツなモデル定義・・・・・

            # 直感的なPO=INVOICEという考え方に戻すべきかも・・・・・
            #
            # modal shiftのmodelingをLT_shiftとの拡張で考える???
            # modal = BOAT/AIR/QURIE
            # LT_shift(modal, w, ,,,,

            for lot in confirmed_S_lots:

                if lot == []:

                    pass

                else:

                    # *********************************************************
                    # child#ship2node = find_node_to_ship(node, lot)
                    # lotidからleaf_nodeのpointerを返す

                    print("lot_ID", lot)

                    leaf_node_name = extract_node_name(lot)

                    print("lot_ID leaf_node_name", lot, leaf_node_name )


                    leaf_node = nodes[leaf_node_name]



                    # 末端からあるnodeAまでleaf_nodeまでのnode_listをpathで返す

                    current_path = []
                    path = []

                    path = find_path_to_leaf_with_parent(node, leaf_node, current_path)

                    # nodes_listを逆にひっくり返す
                    path.reverse()

                    # 出荷先nodeはnodeAの次node、path[1]になる
                    ship2node_name = path[1]

                    ship2node = nodes[ship2node_name]

                    # ここでsupply planを更新している
                    # 出荷先nodeのPSIのPとSに、confirmed_S中のlotをby lotで置く
                    #place_P_in_supply(w, ship2node, lot)
                    place_P_in_supply_LT(w, ship2node, lot)

    for child in node.children:

        feedback_psi_lists(child, nodes)
        #feedback_psi_lists(child, node_psi_dict, nodes)






def copy_P_demand2supply(node): # TOBE 240926
#def update_child_PS(node): # TOBE 240926

    # 明示的に.copyする。
    plan_len = 53 * node.plan_range
    for w in range(0, plan_len):

        node.psi4supply[w][3] = node.psi4demand[w][3].copy()



def PULL_process(node):
    # *******************************************
    # decouple nodeは、pull_Sで出荷指示する
    # *******************************************

    #@241002 childで、親nodeの確定S=確定P=demandのPで計算済み
    # copy S&P demand2supply for PULL
    copy_S_demand2supply(node)
    copy_P_demand2supply(node)


    # 自分のnodeをPS2Iで確定する
    node.calcPS2I4supply()  # calc_psi with PULL_S&P

    print(f"PULL_process applied to {node.name}")



def apply_pull_process(node):

    #@241002 MOVE
    #PULL_process(node)

    for child in node.children:


        PULL_process(child)


        apply_pull_process(child)




def copy_S_demand2supply(node): # TOBE 240926
#def update_child_PS(node): # TOBE 240926

    # 明示的に.copyする。
    plan_len = 53 * node.plan_range
    for w in range(0, plan_len):

        node.psi4supply[w][0] = node.psi4demand[w][0].copy()




def PUSH_process(node):


    # ***************
    # decoupl nodeに入って最初にcalcPS2Iで状態を整える
    # ***************
    node.calcPS2I4supply()  # calc_psi with PULL_S


    # STOP STOP
    ##@241002 decoupling nodeのみpullSで確定ship
    ## *******************************************
    ## decouple nodeは、pull_Sで出荷指示する
    ## *******************************************
    ## copy S demand2supply
    #copy_S_demand2supply(node)
    #
    ## 自分のnodeをPS2Iで確定する
    #node.calcPS2I4supply()  # calc_psi with PUSH_S


    print(f"PUSH_process applied to {node.name}")





def push_pull_all_psi2i_decouple4supply5(node, decouple_nodes):

    if node.name in decouple_nodes:

        # ***************
        # decoupl nodeに入って最初にcalcPS2Iで状態を整える
        # ***************
        node.calcPS2I4supply()  # calc_psi with PULL_S


        #@241002 decoupling nodeのみpullSで確定ship
        # *******************************************
        # decouple nodeは、pull_Sで出荷指示する
        # *******************************************
        copy_S_demand2supply(node)

        PUSH_process(node)         # supply SP2Iしてからの

        apply_pull_process(node)   # demandSに切り替え

    else:

        PUSH_process(node)

        for child in node.children:

            push_pull_all_psi2i_decouple4supply5(child, decouple_nodes)





def map_psi_lots2df(node, D_S_flag, psi_lots):
    if D_S_flag == "demand":
        matrix = node.psi4demand
    elif D_S_flag == "supply":
        matrix = node.psi4supply
    else:
        print("error: wrong D_S_flag is defined")
        return pd.DataFrame()

    for week, row in enumerate(matrix):
        for scoip, lots in enumerate(row):
            for step_no, lot_id in enumerate(lots):
                psi_lots.append([node.name, week, scoip, step_no, lot_id])

    for child in node.children:
        map_psi_lots2df(child, D_S_flag, psi_lots)

    columns = ["node_name", "week", "s-co-i-p", "step_no", "lot_id"]
    df = pd.DataFrame(psi_lots, columns=columns)
    return df





# **************************
# collect_psi_data
# **************************
def collect_psi_data(node, D_S_flag, week_start, week_end, psi_data):
    if D_S_flag == "demand":
        psi_lots = []
        df_demand_plan = map_psi_lots2df(node, D_S_flag, psi_lots)
        df_init = df_demand_plan
    elif D_S_flag == "supply":
        psi_lots = []
        df_supply_plan = map_psi_lots2df(node, D_S_flag, psi_lots)
        df_init = df_supply_plan
    else:
        print("error: D_S_flag should be demand or supply")
        return

    condition1 = df_init["node_name"] == node.name
    condition2 = (df_init["week"] >= week_start) & (df_init["week"] <= week_end)
    df = df_init[condition1 & condition2]

    line_data_2I = df[df["s-co-i-p"].isin([2])]
    bar_data_0S = df[df["s-co-i-p"] == 0]
    bar_data_3P = df[df["s-co-i-p"] == 3]

    line_plot_data_2I = line_data_2I.groupby("week")["lot_id"].count()
    bar_plot_data_3P = bar_data_3P.groupby("week")["lot_id"].count()
    bar_plot_data_0S = bar_data_0S.groupby("week")["lot_id"].count()



    # ノードのREVENUEとPROFITを四捨五入

    # root_out_optからroot_outboundの世界へ変換する
    #@241225 be checked

    #@ STOP
    ##@ TEST node_optとnode_originに、revenueとprofit属性を追加
    #revenue = round(node.revenue)
    #profit  = round(node.profit)


    #@241225 STOP "self.nodes_outbound"がscopeにない
    #node_origin = self.nodes_outbound[node.name]
    #

    revenue = round(node.eval_cs_price_sales_shipped)
    profit = round(node.eval_cs_profit)



    # PROFIT_RATIOを計算して四捨五入
    profit_ratio = round((profit / revenue) * 100, 1) if revenue != 0 else 0

    psi_data.append((node.name, revenue, profit, profit_ratio, line_plot_data_2I, bar_plot_data_3P, bar_plot_data_0S))




# node is "node_opt"
def collect_psi_data_opt(node, node_out, D_S_flag, week_start, week_end, psi_data):
    if D_S_flag == "demand":
        psi_lots = []
        df_demand_plan = map_psi_lots2df(node, D_S_flag, psi_lots)
        df_init = df_demand_plan
    elif D_S_flag == "supply":
        psi_lots = []
        df_supply_plan = map_psi_lots2df(node, D_S_flag, psi_lots)
        df_init = df_supply_plan
    else:
        print("error: D_S_flag should be demand or supply")
        return

    condition1 = df_init["node_name"] == node.name
    condition2 = (df_init["week"] >= week_start) & (df_init["week"] <= week_end)
    df = df_init[condition1 & condition2]

    line_data_2I = df[df["s-co-i-p"].isin([2])]
    bar_data_0S = df[df["s-co-i-p"] == 0]
    bar_data_3P = df[df["s-co-i-p"] == 3]

    line_plot_data_2I = line_data_2I.groupby("week")["lot_id"].count()
    bar_plot_data_3P = bar_data_3P.groupby("week")["lot_id"].count()
    bar_plot_data_0S = bar_data_0S.groupby("week")["lot_id"].count()



    # ノードのREVENUEとPROFITを四捨五入

    # root_out_optからroot_outboundの世界へ変換する
    #@241225 be checked

    #@ STOP
    ##@ TEST node_optとnode_originに、revenueとprofit属性を追加
    #revenue = round(node.revenue)
    #profit  = round(node.profit)


    #@241225 STOP "self.nodes_outbound"がscopeにない
    #node_origin = self.nodes_outbound[node.name]
    #

    # nodeをoptからoutに切り替え
    revenue = round(node_out.eval_cs_price_sales_shipped)
    profit = round(node_out.eval_cs_profit)



    # PROFIT_RATIOを計算して四捨五入
    profit_ratio = round((profit / revenue) * 100, 1) if revenue != 0 else 0

    psi_data.append((node.name, revenue, profit, profit_ratio, line_plot_data_2I, bar_plot_data_3P, bar_plot_data_0S))




#@250110 STOP
## gui/app.py
#class PSIPlannerApp:
#    def __init__(self, root, config):
#        self.root = root
#        self.config = config
#        self.root.title(self.config.APP_NAME)
#
#        self.tree_structure = None
#
#        # 必ず setup_ui を先に呼び出す
#        self.setup_ui()
#        
#        # 必要な初期化処理を後から呼び出す
#        self.initialize_parameters()
#
#
#
#        # PSI planner
#        self.outbound_data = None
#        self.inbound_data = None
#
#        self.root_node_outbound = None
#        self.nodes_outbound = None
#        self.leaf_nodes_out = []
#
#        self.root_node_inbound = None
#        self.nodes_inbound = None
#        self.leaf_nodes_in = []
#
#        self.total_revenue = 0
#        self.total_profit = 0
#        self.profit_ratio = 0
#
#        # View settings
#        self.G = None
#        self.pos_E2E = None
#        self.fig_network = None
#        self.ax_network = None
#
#        # Initialize parameters
#        self.initialize_parameters()




def is_picklable(value):
    try:
        pickle.dumps(value)
    except (pickle.PicklingError, TypeError):
        return False
    return True





class PSIPlannerApp4save:

    #def __init__(self, root):

    def __init__(self):

        #self.root = root
        #self.root.title("Global Weekly PSI Planner")

        self.root_node = None  # root_nodeの定義を追加


        self.lot_size     = 2000      # 初期値

        self.plan_year_st = 2022      # 初期値
        self.plan_range   = 2         # 初期値

        self.pre_proc_LT  = 13        # 初期値 13week = 3month


        self.market_potential = 0     # 初期値 0
        self.target_share     = 0.5   # 初期値 0.5 = 50%
        self.total_supply     = 0     # 初期値 0


        #@ STOP
        #self.setup_ui()

        self.outbound_data = None
        self.inbound_data = None

        # PySI tree
        self.root_node_outbound = None
        self.nodes_outbound     = None
        self.leaf_nodes_out     = []

        self.root_node_inbound  = None
        self.nodes_inbound      = None
        self.leaf_nodes_in      = []

        self.root_node_out_opt  = None
        self.nodes_out_opt      = None
        self.leaf_nodes_opt     = []


        self.optimized_root     = None
        self.optimized_nodes    = None


        # Evaluation on PSI
        self.total_revenue = 0
        self.total_profit  = 0
        self.profit_ratio  = 0


        # view
        self.G = None

        # Optimise
        self.Gdm_structure = None

        self.Gdm = None
        self.Gsp = None

        self.pos_E2E = None

        self.flowDict_opt = {} #None
        self.flowCost_opt = {} #None

        self.total_supply_plan = 0

        # loading files
        self.directory = None
        self.load_directory = None

        self.base_leaf_name = None

        # supply_plan / decoupling / buffer stock
        self.decouple_node_dic = {}

        self.decouple_node_selected = []



    #@ STOP
    #def update_from_psiplannerapp(self, psi_planner_app):
    #    self.__dict__.update(psi_planner_app.__dict__)
    #
    #def update_psiplannerapp(self, psi_planner_app):
    #    psi_planner_app.__dict__.update(self.__dict__)



#@ STOP
#    def update_from_psiplannerapp(self, psi_planner_app):
#        attributes = {key: value for key, value in psi_planner_app.__dict__.items() if key != 'root'}
#        self.__dict__.update(attributes)
#
#    def update_psiplannerapp(self, psi_planner_app):
#        attributes = {key: value for key, value in self.__dict__.items()}
#        psi_planner_app.__dict__.update(attributes)







    def update_from_psiplannerapp(self, psi_planner_app):
        attributes = {key: value for key, value in psi_planner_app.__dict__.items()
                      if key != 'root' and is_picklable(value) and not isinstance(value, (tk.Tk, tk.Widget, tk.Toplevel, tk.Variable))}
        self.__dict__.update(attributes)

    def update_psiplannerapp(self, psi_planner_app):
        attributes = {key: value for key, value in self.__dict__.items()}
        psi_planner_app.__dict__.update(attributes)




# **************************
# cost_stracture
# **************************




def make_stack_bar4cost_stracture(cost_dict):
    attributes_B = [
        'cs_direct_materials_costs',
        'cs_marketing_promotion',
        'cs_sales_admin_cost',
        'cs_tax_portion',
        'cs_logistics_costs',
        'cs_warehouse_cost',
        'cs_prod_indirect_labor',
        'cs_prod_indirect_others',
        'cs_direct_labor_costs',
        'cs_depreciation_others',
        'cs_profit',
    ]

    colors = {
        'cs_direct_materials_costs': 'lightgray',
        'cs_marketing_promotion': 'darkblue',
        'cs_sales_admin_cost': 'blue',
        'cs_tax_portion': 'gray',
        'cs_logistics_costs': 'cyan',
        'cs_warehouse_cost': 'magenta',
        'cs_prod_indirect_labor': 'green',
        'cs_prod_indirect_others': 'lightgreen',
        'cs_direct_labor_costs': 'limegreen',
        'cs_depreciation_others': 'yellowgreen',
        'cs_profit': 'gold',
    }

    nodes = list(cost_dict.keys())
    bar_width = 0.5

    # Initialize the bottom of the bars
    bottoms = np.zeros(len(nodes))

    fig, ax = plt.subplots()

    for attr in attributes_B:
        values = [cost_dict[node][attr] for node in cost_dict]
        ax.bar(nodes, values, bar_width, label=attr, color=colors[attr], bottom=bottoms)
        bottoms += values

        # Add text on bars
        for i, value in enumerate(values):
            if value > 0:
                ax.text(i, bottoms[i] - value / 2, f'{value:.1f}', ha='center', va='center', fontsize=8, color='white')

    # Add total values on top of bars
    total_values = [sum(cost_dict[node][attr] for attr in attributes_B) for node in cost_dict]
    for i, total in enumerate(total_values):
        ax.text(i, total + 2, f'{total:.1f}', ha='center', va='bottom', fontsize=8)
        #ax.text(i, total + 2, f'{total:.1f}', ha='center', va='bottom', fontsize=10)

    ax.set_title('Supply Chain Cost Structure on Common Planning Unit')
    ax.set_xlabel('Node')
    ax.set_ylabel('Amount')

    # Set legend with smaller fontsize
    ax.legend(title='Attribute', fontsize=6 )
    #ax.legend(title='Attribute', fontsize='small')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




def make_stack_bar4cost_stracture_OLD(cost_dict):
    attributes_B = [
        'cs_direct_materials_costs',
        'cs_marketing_promotion',
        'cs_sales_admin_cost',
        'cs_tax_portion',
        'cs_logistics_costs',
        'cs_warehouse_cost',
        'cs_prod_indirect_labor',
        'cs_prod_indirect_others',
        'cs_direct_labor_costs',
        'cs_depreciation_others',
        'cs_profit',
    ]

    colors = {
        'cs_direct_materials_costs': 'lightgray',
        'cs_marketing_promotion': 'darkblue',
        'cs_sales_admin_cost': 'blue',
        'cs_tax_portion': 'gray',
        'cs_logistics_costs': 'cyan',
        'cs_warehouse_cost': 'magenta',
        'cs_prod_indirect_labor': 'green',
        'cs_prod_indirect_others': 'lightgreen',
        'cs_direct_labor_costs': 'limegreen',
        'cs_depreciation_others': 'yellowgreen',
        'cs_profit': 'gold',
    }

    nodes = list(cost_dict.keys())
    bar_width = 0.5

    # Initialize the bottom of the bars
    bottoms = np.zeros(len(nodes))

    fig, ax = plt.subplots()

    for attr in attributes_B:
        values = [cost_dict[node][attr] for node in cost_dict]
        ax.bar(nodes, values, bar_width, label=attr, color=colors[attr], bottom=bottoms)
        bottoms += values

        # Add text on bars
        for i, value in enumerate(values):
            if value > 0:
                ax.text(i, bottoms[i] - value / 2, f'{value:.1f}', ha='center', va='center', fontsize=8, color='white')

    # Add total values on top of bars
    total_values = [sum(cost_dict[node][attr] for attr in attributes_B) for node in cost_dict]
    for i, total in enumerate(total_values):
        ax.text(i, total + 2, f'{total:.1f}', ha='center', va='bottom', fontsize=10)

    ax.set_title('Supply Chain Cost Structure on Common Planning Unit')
    ax.set_xlabel('Node')
    ax.set_ylabel('Amount')
    ax.legend(title='Attribute')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




# gui/app.py
class PSIPlannerApp:
    def __init__(self, root, config):
    #def __init__(self, root):

        self.root = root
        self.config = config
        self.root.title(self.config.APP_NAME)
        
        self.tree_structure = None

        # 必ず setup_ui を先に呼び出す
        self.setup_ui()
        
        # 必要な初期化処理を後から呼び出す
        self.initialize_parameters()

        #@ STOP moved to config.py
        #self.lot_size     = 2000      # 初期値
        #self.plan_year_st = 2022      # 初期値
        #self.plan_range   = 2         # 初期値
        #self.pre_proc_LT  = 13        # 初期値 13week = 3month
        #self.market_potential = 0     # 初期値 0
        #self.target_share     = 0.5   # 初期値 0.5 = 50%
        #self.total_supply     = 0     # 初期値 0


        # ********************************
        # PSI planner
        # ********************************
        self.outbound_data = None
        self.inbound_data = None

        # PySI tree
        self.root_node_outbound = None
        self.nodes_outbound     = None
        self.leaf_nodes_out     = []

        self.root_node_inbound  = None
        self.nodes_inbound      = None
        self.leaf_nodes_in      = []

        self.root_node_out_opt  = None
        self.nodes_out_opt      = None
        self.leaf_nodes_opt     = []


        self.optimized_root     = None
        self.optimized_nodes    = None


        # Evaluation on PSI
        self.total_revenue = 0
        self.total_profit  = 0
        self.profit_ratio  = 0

        # view
        self.G = None

        # Optimise
        self.Gdm_structure = None

        self.Gdm = None
        self.Gsp = None

        self.pos_E2E = None

        self.flowDict_opt = {} #None
        self.flowCost_opt = {} #None

        self.total_supply_plan = 0

        # loading files
        self.directory = None
        self.load_directory = None

        self.base_leaf_name = None

        # supply_plan / decoupling / buffer stock
        self.decouple_node_dic = {}

        self.decouple_node_selected = []






    def setup_ui(self):

        print("setup_ui is processing")

        # フォントの設定
        custom_font = tkfont.Font(family="Helvetica", size=12)

        # メニュー全体のフォントサイズを設定
        self.root.option_add('*TearOffMenu*Font', custom_font)
        self.root.option_add('*Menu*Font', custom_font)

        # メニューバーの作成
        menubar = tk.Menu(self.root)

        # スタイルの設定
        style = ttk.Style()
        style.configure("TMenubutton", font=("Helvetica", 12))

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="OPEN: select Directory", command=self.load_data_files)
        file_menu.add_separator()
        file_menu.add_command(label="SAVE: to Directory", command=self.save_to_directory)

        file_menu.add_command(label="LOAD: from Directory", command=self.load_from_directory)

        file_menu.add_separator()
        file_menu.add_command(label="EXIT", command=self.on_exit)
        menubar.add_cascade(label=" FILE  ", menu=file_menu)

        # Optimize Parameter menu
        optimize_menu = tk.Menu(menubar, tearoff=0)
        optimize_menu.add_command(label="Weight: Cost Stracture on Common Plan Unit", command=self.show_cost_stracture_bar_graph)
        optimize_menu.add_command(label="Capacity: Market Demand", command=self.show_month_data_csv)
        menubar.add_cascade(label="Optimize Parameter", menu=optimize_menu)



        # Report menu
        report_menu = tk.Menu(menubar, tearoff=0)

        report_menu.add_command(label="Outbound: PSI to csv file", command=self.outbound_psi_to_csv)

        report_menu.add_command(label="Outbound: Lot by Lot data to csv", command=self.outbound_lot_by_lot_to_csv)

        report_menu.add_separator()

        report_menu.add_command(label="Inbound: PSI to csv file", command=self.inbound_psi_to_csv)
        report_menu.add_command(label="Inbound: Lot by Lot data to csv", command=self.inbound_lot_by_lot_to_csv)

        report_menu.add_separator()

        report_menu.add_separator()

        report_menu.add_command(label="Value Chain: Cost Stracture a Lot", command=self.lot_cost_structure_to_csv)

        report_menu.add_command(label="Supply Chain: Revenue Profit", command=self.supplychain_performance_to_csv)



        #report_menu.add_separator()
        #
        #report_menu.add_command(label="PSI for Excel", command=self.psi_for_excel)


        menubar.add_cascade(label="Report", menu=report_menu)




        # 3D overview menu
        overview_menu = tk.Menu(menubar, tearoff=0)
        overview_menu.add_command(label="3D overview on Lots based Plan", command=self.show_3d_overview)
        menubar.add_cascade(label="3D overview", menu=overview_menu)

        self.root.config(menu=menubar)

        # フレームの作成
        self.frame = ttk.Frame(self.root)
        self.frame.pack(side=tk.LEFT, fill=tk.Y)

        # Lot size entry
        self.lot_size_label = ttk.Label(self.frame, text="Lot Size:")
        self.lot_size_label.pack(side=tk.TOP)
        self.lot_size_entry = ttk.Entry(self.frame, width=10)
        self.lot_size_entry.pack(side=tk.TOP)

        #@250117 UPDATE
        self.lot_size_entry.insert(0, str(self.config.DEFAULT_LOT_SIZE))  # 初期値を設定
        #self.lot_size_entry.insert(0, str(self.lot_size))  # 初期値を設定

        # Plan Year Start entry
        self.plan_year_label = ttk.Label(self.frame, text="Plan Year Start:")
        self.plan_year_label.pack(side=tk.TOP)
        self.plan_year_entry = ttk.Entry(self.frame, width=10)
        self.plan_year_entry.pack(side=tk.TOP)


        self.plan_year_entry.insert(0, str(self.config.DEFAULT_START_YEAR))  # 初期値を設定
        #self.plan_year_entry.insert(0, str(self.plan_year_st))  # 初期値を設定

        # Plan Range entry
        self.plan_range_label = ttk.Label(self.frame, text="Plan Range:")
        self.plan_range_label.pack(side=tk.TOP)
        self.plan_range_entry = ttk.Entry(self.frame, width=10)
        self.plan_range_entry.pack(side=tk.TOP)


        self.plan_range_entry.insert(0, str(self.config.DEFAULT_PLAN_RANGE))  # 初期値を設定
        #self.plan_range_entry.insert(0, str(self.plan_range))  # 初期値を設定

        # 1行分の空白を追加
        self.space_label = ttk.Label(self.frame, text="")
        self.space_label.pack(side=tk.TOP)



        #@250120 RUN
        # Demand Planning ボタン（グレイアウト）
        self.Demand_Pl_button = ttk.Button(
            self.frame,
            text="Demand Planning",
            command=lambda: None,  # 無効化
            state="disabled",  # ボタンを無効化
            style="Disabled.TButton"  # スタイルを適用
        )
        self.Demand_Pl_button.pack(side=tk.TOP)


        #@250120 STOP
        ## Demand Planning buttons
        #self.Demand_Pl_button = ttk.Button(self.frame, text="Demand Planning", command=self.demand_planning)
        #self.Demand_Pl_button.pack(side=tk.TOP)




        # Plan Year Start entry
        self.pre_proc_LT_label = ttk.Label(self.frame, text="pre_proc_LT:")
        self.pre_proc_LT_label.pack(side=tk.TOP)
        self.pre_proc_LT_entry = ttk.Entry(self.frame, width=10)
        self.pre_proc_LT_entry.pack(side=tk.TOP)


        self.pre_proc_LT_entry.insert(0, str(self.config.DEFAULT_PRE_PROC_LT))  # 初期値を設定
        #self.pre_proc_LT_entry.insert(0, str(self.pre_proc_LT))  # 初期値を設定


        #@250120 RUN
        # Demand Leveling ボタン（グレイアウト）
        self.Demand_Lv_button = ttk.Button(
            self.frame,
            text="Demand Leveling",
            command=lambda: None,  # 無効化
            state="disabled",  # ボタンを無効化
            style="Disabled.TButton"  # スタイルを適用
        )
        self.Demand_Lv_button.pack(side=tk.TOP)


        #@250120 STOP
        ## Demand Leveling button
        #self.Demand_Lv_button = ttk.Button(self.frame, text="Demand Leveling", command=self.demand_leveling)
        #self.Demand_Lv_button.pack(side=tk.TOP)





        # add a blank line
        self.space_label = ttk.Label(self.frame, text="")
        self.space_label.pack(side=tk.TOP)

        # Supply Planning button
        self.supply_planning_button = ttk.Button(self.frame, text="Supply Planning ", command=self.supply_planning)
        self.supply_planning_button.pack(side=tk.TOP)

        # add a blank line
        self.space_label = ttk.Label(self.frame, text="")
        self.space_label.pack(side=tk.TOP)

        # Eval_buffer_stock buttons
        self.eval_buffer_stock_button = ttk.Button(self.frame, text="Eval Buffer Stock ", command=self.eval_buffer_stock)
        self.eval_buffer_stock_button.pack(side=tk.TOP)

        # add a blank line
        self.space_label = ttk.Label(self.frame, text="")
        self.space_label.pack(side=tk.TOP)

        # Optimize Network button
        self.optimize_button = ttk.Button(self.frame, text="OPT Supply Alloc", command=self.optimize_network)
        self.optimize_button.pack(side=tk.TOP)

        # Plot area divided into two frames
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Network Graph frame
        self.network_frame = ttk.Frame(self.plot_frame)
        self.network_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # New Frame for Parameters at the top of the network_frame
        self.param_frame = ttk.Frame(self.network_frame)
        self.param_frame.pack(side=tk.TOP, fill=tk.X)


        # Global Market Potential, Target Share, Total Supply Plan input fields arranged horizontally
        self.gmp_label = tk.Label(self.param_frame, text="Market Potential:", background='navy', foreground='white', font=('Helvetica', 10, 'bold'))
        self.gmp_label.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=10)
        self.gmp_entry = tk.Entry(self.param_frame, width=10)
        self.gmp_entry.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=10)

        self.ts_label = tk.Label(self.param_frame, text="TargetShare(%)", background='navy', foreground='white', font=('Helvetica', 10, 'bold'))
        self.ts_label.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=10)
        self.ts_entry = tk.Entry(self.param_frame, width=5)
        self.ts_entry.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=10)



        self.ts_entry.insert(0, self.config.DEFAULT_TARGET_SHARE * 100) # 初期値
        #self.ts_entry.insert(0, self.target_share * 100) # 初期値

        self.tsp_label = tk.Label(self.param_frame, text="Total Supply:", background='navy', foreground='white', font=('Helvetica', 10, 'bold'))
        self.tsp_label.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=10)
        self.tsp_entry = tk.Entry(self.param_frame, width=10)
        self.tsp_entry.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=10)
        self.tsp_entry.config(bg='lightgrey')  # 背景色をlightgreyに設定




        # イベントバインディング
        self.gmp_entry.bind("<Return>", self.update_total_supply_plan)
        self.ts_entry.bind("<Return>", self.update_total_supply_plan)

        self.fig_network, self.ax_network = plt.subplots(figsize=(4, 8))  # 横幅を縮小
        self.canvas_network = FigureCanvasTkAgg(self.fig_network, master=self.network_frame)
        self.canvas_network.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig_network.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Evaluation result area
        self.eval_frame = ttk.Frame(self.plot_frame)
        self.eval_frame.pack(side=tk.TOP, fill=tk.X, padx=(20, 0))  # 横方向に配置

        # Total Revenue
        self.total_revenue_label = ttk.Label(self.eval_frame, text="Total Revenue:", background='darkgreen', foreground='white', font=('Helvetica', 10, 'bold'))
        self.total_revenue_label.pack(side=tk.LEFT, padx=5, pady=10)
        self.total_revenue_entry = ttk.Entry(self.eval_frame, width=10, state='readonly')
        self.total_revenue_entry.pack(side=tk.LEFT, padx=5, pady=10)

        # Total Profit
        self.total_profit_label = ttk.Label(self.eval_frame, text="Total Profit:", background='darkgreen', foreground='white', font=('Helvetica', 10, 'bold'))
        self.total_profit_label.pack(side=tk.LEFT, padx=5, pady=10)
        self.total_profit_entry = ttk.Entry(self.eval_frame, width=10, state='readonly')
        self.total_profit_entry.pack(side=tk.LEFT, padx=5, pady=10)



        # Profit Ratio
        self.profit_ratio_label = ttk.Label(self.eval_frame, text="Profit Ratio:", background='darkgreen', foreground='white', font=('Helvetica', 10, 'bold'))
        self.profit_ratio_label.pack(side=tk.LEFT, padx=5, pady=10)
        self.profit_ratio_entry = ttk.Entry(self.eval_frame, width=10, state='readonly')
        self.profit_ratio_entry.pack(side=tk.LEFT, padx=5, pady=10)

        # PSI Graph scroll frame (moved to below evaluation area)
        self.psi_frame = ttk.Frame(self.plot_frame)
        self.psi_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas_psi = tk.Canvas(self.psi_frame)
        self.scrollbar = ttk.Scrollbar(self.psi_frame, orient="vertical", command=self.canvas_psi.yview)
        self.scrollable_frame = ttk.Frame(self.canvas_psi)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_psi.configure(
                scrollregion=self.canvas_psi.bbox("all")
            )
        )

        self.canvas_psi.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas_psi.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_psi.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        #@250120 STOP
        ## 初期化関数を呼び出してパラメータ設定
        #self.initialize_parameters()




    def update_total_supply_plan(self, event):
        try:
            market_potential = float(self.gmp_entry.get().replace(',', ''))
            target_share = float(self.ts_entry.get().replace('%', ''))/100
        except ValueError:
            print("Invalid input for Global Market Potential or Target Share.")
            return

        # Total Supply Planの再計算
        total_supply_plan = round(market_potential * target_share)

        self.total_supply_plan = total_supply_plan

        # Total Supply Planフィールドの更新
        self.tsp_entry.config(state='normal')
        self.tsp_entry.delete(0, tk.END)
        self.tsp_entry.insert(0, "{:,}".format(total_supply_plan))  # 3桁毎にカンマ区切りで表示
        self.tsp_entry.config(state='normal')



    def initialize_parameters(self):

        print("Initializing parameters")
        self.lot_size     = self.config.DEFAULT_LOT_SIZE
        self.plan_year_st = self.config.DEFAULT_START_YEAR
        self.plan_range   = self.config.DEFAULT_PLAN_RANGE

        self.pre_proc_LT  = self.config.DEFAULT_PRE_PROC_LT
    
        # self.market_potential = 0 # initial setting from "demand_generate"
        self.target_share = self.config.DEFAULT_TARGET_SHARE
        self.total_supply = 0










        if not hasattr(self, 'gmp_entry') or not hasattr(self, 'ts_entry') or not hasattr(self, 'tsp_entry'):
            raise AttributeError("Required UI components (gmp_entry, ts_entry, tsp_entry) have not been initialized.")


        print("Setting market potential and share")
        # Calculation and setting of Global Market Potential
        market_potential = getattr(self, 'market_potential', self.config.DEFAULT_MARKET_POTENTIAL)  # Including initial settings

        self.gmp_entry.delete(0, tk.END)
        self.gmp_entry.insert(0, "{:,}".format(market_potential))  # Display with comma separated thousands

        # Initial setting of Target Share (already set in setup_ui)

        # Calculation and setting of Total Supply Plan
        target_share = float(self.ts_entry.get().replace('%', ''))/100  # Convert string to float and remove %

        total_supply_plan = round(market_potential * target_share)
        self.tsp_entry.delete(0, tk.END)
        self.tsp_entry.insert(0, "{:,}".format(total_supply_plan))  # Display with comma separated thousands

        #self.global_market_potential  = global_market_potential

        self.market_potential         = market_potential
        self.target_share             = target_share           
        self.total_supply_plan        = total_supply_plan
        print(f"At initialization - market_potential: {self.market_potential}, target_share: {self.target_share}")  # Add log





    def updated_parameters(self):
        print(f"updated_parameters更新前 - market_potential: {self.market_potential}, target_share: {self.target_share}")  # ログ追加

        # Market Potentialの計算と設定
        market_potential = self.market_potential
        print("market_potential", market_potential)
        
        self.gmp_entry.delete(0, tk.END)
        self.gmp_entry.insert(0, "{:,}".format(market_potential))  # 3桁毎にカンマ区切りで表示

        # Target Shareの初期値設定（すでにsetup_uiで設定済み）
        #@ ADD: Keep the current target_share value if user has not entered a new value
        if self.ts_entry.get():
            target_share = float(self.ts_entry.get().replace('%', '')) / 100  # 文字列を浮動小数点数に変換して%を除去
        else:
            target_share = self.target_share

        # Total Supply Planの計算と設定
        total_supply_plan = round(market_potential * target_share)
        self.tsp_entry.delete(0, tk.END)
        self.tsp_entry.insert(0, "{:,}".format(total_supply_plan))  # 3桁毎にカンマ区切りで表示

        self.market_potential = market_potential
        self.target_share = target_share
        self.total_supply_plan = total_supply_plan

        print(f"updated_parameters更新時 - market_potential: {self.market_potential}, target_share: {self.target_share}")  # ログ追加





# ******************************
# actions
# ******************************



    def load_data_files(self):
        directory = filedialog.askdirectory(title="Select Data Directory")

        if directory:
            try:
                self.lot_size = int(self.lot_size_entry.get())
                self.plan_year_st = int(self.plan_year_entry.get())
                self.plan_range = int(self.plan_range_entry.get())
            except ValueError:
                print("Invalid input for lot size, plan year start, or plan range. Using default values.")

            self.outbound_data = []
            self.inbound_data = []

            data_file_list = os.listdir(directory)

            self.directory = directory
            self.load_directory = directory

            if "profile_tree_outbound.csv" in data_file_list:
                file_path = os.path.join(directory, "profile_tree_outbound.csv")
                nodes_outbound = {}
                nodes_outbound, root_node_name_out = create_tree_set_attribute(file_path)
                root_node_outbound = nodes_outbound[root_node_name_out]

                def make_leaf_nodes(node, leaf_list):
                    if not node.children:
                        leaf_list.append(node.name)
                    for child in node.children:
                        make_leaf_nodes(child, leaf_list)
                    return leaf_list

                leaf_nodes_out = make_leaf_nodes(root_node_outbound, [])
                self.nodes_outbound = nodes_outbound
                self.root_node_outbound = root_node_outbound
                self.leaf_nodes_out = leaf_nodes_out
                set_positions(root_node_outbound)
                set_parent_all(root_node_outbound)
                print_parent_all(root_node_outbound)
            else:
                print("error: profile_tree_outbound.csv is missed")

            if "profile_tree_inbound.csv" in data_file_list:
                file_path = os.path.join(directory, "profile_tree_inbound.csv")
                nodes_inbound = {}
                nodes_inbound, root_node_name_in = create_tree_set_attribute(file_path)
                root_node_inbound = nodes_inbound[root_node_name_in]
                self.nodes_inbound = nodes_inbound
                self.root_node_inbound = root_node_inbound
                set_positions(root_node_inbound)
                set_parent_all(root_node_inbound)
                print_parent_all(root_node_inbound)
            else:
                print("error: profile_tree_inbound.csv is missed")

            if "node_cost_table_outbound.csv" in data_file_list:
                file_path = os.path.join(directory, "node_cost_table_outbound.csv")
                read_set_cost(file_path, nodes_outbound)
            else:
                print("error: node_cost_table_outbound.csv is missed")

            if "node_cost_table_inbound.csv" in data_file_list:
                file_path = os.path.join(directory, "node_cost_table_inbound.csv")
                read_set_cost(file_path, nodes_inbound)
            else:
                print("error: node_cost_table_inbound.csv is missed")

            if "S_month_data.csv" in data_file_list:
                in_file_path = os.path.join(directory, "S_month_data.csv")

                df_weekly, plan_range, plan_year_st = process_monthly_demand(in_file_path, self.lot_size)

                #df_weekly, plan_range, plan_year_st = trans_month2week2lot_id_list(in_file_path, self.lot_size)

                self.plan_year_st = plan_year_st
                self.plan_range = plan_range
                self.plan_year_entry.delete(0, tk.END)
                self.plan_year_entry.insert(0, str(self.plan_year_st))
                self.plan_range_entry.delete(0, tk.END)
                self.plan_range_entry.insert(0, str(self.plan_range))
                out_file_path = os.path.join(directory, "S_iso_week_data.csv")
                df_weekly.to_csv(out_file_path, index=False)

            else:
                print("error: S_month_data.csv is missed")

            root_node_outbound.set_plan_range_lot_counts(plan_range, plan_year_st)
            root_node_inbound.set_plan_range_lot_counts(plan_range, plan_year_st)

            node_psi_dict_Ot4Dm = make_psi_space_dict(root_node_outbound, {}, plan_range)
            node_psi_dict_Ot4Sp = make_psi_space_dict(root_node_outbound, {}, plan_range)
            node_psi_dict_In4Dm = make_psi_space_dict(root_node_inbound, {}, plan_range)
            node_psi_dict_In4Sp = make_psi_space_dict(root_node_inbound, {}, plan_range)
            set_dict2tree_psi(root_node_outbound, "psi4demand", node_psi_dict_Ot4Dm)
            set_dict2tree_psi(root_node_outbound, "psi4supply", node_psi_dict_Ot4Sp)
            set_dict2tree_psi(root_node_inbound, "psi4demand", node_psi_dict_In4Dm)
            set_dict2tree_psi(root_node_inbound, "psi4supply", node_psi_dict_In4Sp)



            #set_df_Slots2psi4demand(self.root_node_outbound, df_weekly)
            set_df_Slots2psi4demand(root_node_outbound, df_weekly)

            print("Data files loaded successfully.")




# **** A PART of ORIGINAL load_data_files *****

        def count_lots_on_S_psi4demand(node, S_list):
                if not node.children:
                        for w_psi in node.psi4demand:
                                S_list.append(w_psi[0])
                for child in node.children:
                        count_lots_on_S_psi4demand(child, S_list)
                return S_list

        S_list = []
        year_lots_list4S = []
        S_list = count_lots_on_S_psi4demand(root_node_outbound, S_list)

        #@250117 STOP
        #plan_year_st = year_st

        for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):
                year_lots4S = count_lots_yyyy(S_list, str(yyyy))
                year_lots_list4S.append(year_lots4S)

        self.market_potential = year_lots_list4S[1]
        print("self.market_potential", self.market_potential)

        self.total_supply_plan = round(self.market_potential * self.target_share)
        print("self.total_supply_plan", self.total_supply_plan)

        for filename in os.listdir(directory):
                if filename.endswith(".csv"):
                        filepath = os.path.join(directory, filename)
                        print(f"Loading file: {filename}")
                        if "outbound" in filename.lower():
                                self.outbound_data.append(pd.read_csv(filepath))
                        elif "inbound" in filename.lower():
                                self.inbound_data.append(pd.read_csv(filepath))
        print("Outbound files loaded.")
        print("Inbound files loaded.")

        def find_node_with_cost_standard_flag(nodes, flag_value):
                for node_name, node in nodes.items():
                        if node.cost_standard_flag == flag_value:
                                return node_name, node
                return None, None

        node_name, base_leaf = find_node_with_cost_standard_flag(nodes_outbound, 100)
        self.base_leaf_name = node_name

        if node_name is None:
                print("NO cost_standard = 100 in profile")
        else:
                print(f"Node name: {node_name}, Base leaf: {base_leaf}")

        root_price = set_price_leaf2root(base_leaf, self.root_node_outbound, 100)
        print("root_price", root_price)
        set_value_chain_outbound(root_price, self.root_node_outbound)



        print("demand_planning execute")




        calc_all_psi2i4demand(self.root_node_outbound)

        self.update_evaluation_results()
        self.decouple_node_selected = []
        self.view_nx_matlib_stop_draw()



        print("Demand Leveling execute")
        year_st = self.plan_year_st
        year_end = year_st + self.plan_range - 1
        pre_prod_week = self.config.DEFAULT_PRE_PROC_LT
        #pre_prod_week = self.pre_proc_LT

        demand_leveling_on_ship(self.root_node_outbound, pre_prod_week, year_st, year_end)

        self.root_node_outbound.calcS2P_4supply()
        self.root_node_outbound.calcPS2I4supply()
        feedback_psi_lists(self.root_node_outbound, self.nodes_outbound)

        self.update_evaluation_results()
        self.psi_backup_to_file(self.root_node_outbound, 'psi_backup.pkl')
        self.view_nx_matlib_stop_draw()



        print("Supply planning with Decoupling points")
        self.root_node_outbound = self.psi_restore_from_file('psi_backup.pkl')

        if not self.decouple_node_selected:
                nodes_decouple_all = make_nodes_decouple_all(self.root_node_outbound)
                print("nodes_decouple_all", nodes_decouple_all)
                decouple_node_names = nodes_decouple_all[-2]
        else:
                decouple_node_names = self.decouple_node_selected

        push_pull_all_psi2i_decouple4supply5(self.root_node_outbound, decouple_node_names)


        # eval area
        self.update_evaluation_results()


        # network area
        self.decouple_node_selected = decouple_node_names
        self.view_nx_matlib4opt()



        # PSI area
        self.root.after(1000, self.show_psi("outbound", "supply"))



        # ****************************
        # market potential Graph viewing
        # ****************************
        self.initialize_parameters()




        # Enable buttons after loading is complete
        self.supply_planning_button.config(state="normal")
        self.eval_buffer_stock_button.config(state="normal")
        print("Data files loaded and buttons enabled.")




        #try:
        #    # Perform data-loading steps
        #    self.root.update_idletasks()  # Ensure the GUI is updated during the process
        #
        #    # Assume data is successfully loaded
        #    print("Data loaded successfully!")
        #
        #    # Re-enable buttons
        #    self.supply_planning_button.config(state="normal")
        #    self.eval_buffer_stock_button.config(state="normal")
        #
        #except Exception as e:
        #    print(f"Error during data loading: {e}")
        #    tk.messagebox.showerror("Error", "Failed to load data files.")


        # Return focus to the main window
        self.root.focus_force()






# **** A PART of ORIGINAL load_data_files END *****


# **** call_backs *****


    def save_data(self, save_directory):

        print(f"保存前 - market_potential: {self.market_potential}, target_share: {self.target_share}")  # ログ追加

        print(f"保存前 - total_revenue : {self.total_revenue}, total_profit : {self.total_profit}")  

        psi_planner_app_save = PSIPlannerApp4save()
        psi_planner_app_save.update_from_psiplannerapp(self)

        print(f"保存時 - market_potential: {psi_planner_app_save.market_potential}, target_share: {psi_planner_app_save.target_share}")  # ログ追加

        print(f"保存時 - total_revenue: {psi_planner_app_save.total_revenue}, total_profit: {psi_planner_app_save.total_profit}")  




        with open(os.path.join(save_directory, 'psi_planner_app.pkl'), "wb") as f:
            pickle.dump(psi_planner_app_save.__dict__, f)
        print("データを保存しました。")




    def save_to_directory(self):
        # 1. Save先となるdirectoryの問い合わせ
        save_directory = filedialog.askdirectory()

        if not save_directory:
            return  # ユーザーがキャンセルした場合

        # 2. 初期処理のcsv fileのコピー
        for filename in os.listdir(self.directory):
            if filename.endswith('.csv'):
                full_file_name = os.path.join(self.directory, filename)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, save_directory)


        # 3. Tree構造の保存
        with open(os.path.join(save_directory, 'root_node_outbound.pkl'), 'wb') as f:
            pickle.dump(self.root_node_outbound, f)
            print(f"root_node_outbound saved: {self.root_node_outbound}")

        with open(os.path.join(save_directory, 'root_node_inbound.pkl'), 'wb') as f:
            pickle.dump(self.root_node_inbound, f)
            print(f"root_node_inbound saved: {self.root_node_inbound}")

        with open(os.path.join(save_directory, 'root_node_out_opt.pkl'), 'wb') as f:
            pickle.dump(self.root_node_out_opt, f)
            print(f"root_node_out_opt saved: {self.root_node_out_opt}")


        # 4. グラフデータの保存
        nx.write_gml(self.G, f"{save_directory}/G.gml")
        nx.write_gml(self.Gdm_structure, f"{save_directory}/Gdm_structure.gml")
        nx.write_gml(self.Gsp, f"{save_directory}/Gsp.gml")
        print(f"グラフが{save_directory}に保存されました")

        nx.write_gpickle(self.G, os.path.join(save_directory, 'G.gpickle'))
        nx.write_gpickle(self.Gdm_structure, os.path.join(save_directory, 'Gdm_structure.gpickle'))
        nx.write_gpickle(self.Gsp, os.path.join(save_directory, 'Gsp.gpickle'))
        print("Graph data saved.")



        # saveの前にself.market_potential,,,をupdate

        #self.initialize_parameters()
        self.updated_parameters()

        # 5. PSIPlannerAppのデータ・インスタンスの保存
        self.save_data(save_directory)

        # 追加：ファイルの存在とサイズの確認
        for filename in ['root_node_outbound.pkl', 'root_node_inbound.pkl', 'psi_planner_app.pkl']:
            full_file_name = os.path.join(save_directory, filename)
            if os.path.exists(full_file_name):
                file_size = os.path.getsize(full_file_name)
                print(f"{filename} exists, size: {file_size} bytes")
            else:
                print(f"{filename} does not exist")

        # 6. 完了メッセージの表示
        messagebox.showinfo("Save Completed", "Plan data save is completed")





    def load_data(self, load_directory):
        with open(os.path.join(load_directory, 'psi_planner_app.pkl'), "rb") as f:
            loaded_attributes = pickle.load(f)

    #@ STOP this is a sample code for "fixed file"
    #def load_data(self, filename="saved_data.pkl"):
    #    with open(filename, "rb") as f:
    #        loaded_attributes = pickle.load(f)


        psi_planner_app_save = PSIPlannerApp4save()
        psi_planner_app_save.__dict__.update(loaded_attributes)
        
        # 選択的にインスタンス変数を更新
        self.root_node = psi_planner_app_save.root_node


        #@ STOP
        #self.D_S_flag = psi_planner_app_save.D_S_flag
        #self.week_start = psi_planner_app_save.week_start
        #self.week_end = psi_planner_app_save.week_end

        self.decouple_node_selected=psi_planner_app_save.decouple_node_selected


        self.G = psi_planner_app_save.G
        self.Gdm = psi_planner_app_save.Gdm
        self.Gdm_structure = psi_planner_app_save.Gdm_structure
        self.Gsp = psi_planner_app_save.Gsp
        self.pos_E2E = psi_planner_app_save.pos_E2E

        self.total_revenue = psi_planner_app_save.total_revenue
        print("load_data: self.total_revenue", self.total_revenue)
        self.total_profit = psi_planner_app_save.total_profit
        print("load_data: self.total_profit", self.total_profit)

        self.flowDict_opt = psi_planner_app_save.flowDict_opt
        self.flowCost_opt = psi_planner_app_save.flowCost_opt


        self.market_potential = psi_planner_app_save.market_potential
        print("self.market_potential", self.market_potential)

        self.target_share = psi_planner_app_save.target_share
        print("self.target_share", self.target_share)

        # エントリウィジェットに反映する
        self.ts_entry.delete(0, tk.END)
        self.ts_entry.insert(0, f"{self.target_share * 100:.0f}")  # 保存された値を反映



        print(f"読み込み時 - market_potential: {self.market_potential}, target_share: {self.target_share}")  # ログ追加
        print("データをロードしました。")





    def regenerate_nodes(self, root_node):
        nodes = {}

        def traverse(node):
            nodes[node.name] = node
            for child in node.children:
                traverse(child)

        traverse(root_node)
        return nodes





    def load_from_directory(self):
        # 1. Load元となるdirectoryの問い合わせ
        load_directory = filedialog.askdirectory()

        if not load_directory:
            return  # ユーザーがキャンセルした場合

        # 2. Tree構造の読み込み
        self.load_directory = load_directory
        self.directory      = load_directory # for "optimized network"
        self._load_tree_structure(load_directory)








        # 3. PSIPlannerAppのデータ・インスタンスの読み込み
        self.load_data(load_directory)

        # if "save files" are NOT optimized one
        if os.path.exists(f"{load_directory}/root_node_out_opt.pkl"):
            pass
        else:
            self.flowDict_opt = {}  # NO optimize



        ## 3. PSIPlannerAppのデータ・インスタンスの読み込みと更新
        #self.selective_update(load_directory)


        # 4. nodes_outboundとnodes_inboundを再生成
        self.nodes_outbound = self.regenerate_nodes(self.root_node_outbound)
        self.nodes_inbound = self.regenerate_nodes(self.root_node_inbound)

        #self.nodes_out_opt = self.regenerate_nodes(self.root_node_out_opt)


        print("load_from_directory self.decouple_node_selected", self.decouple_node_selected)





        #@241224 ADD
        # eval area
        self.update_evaluation_results()


        ## 5. ネットワークグラフの描画
        #self.draw_networkx_graph()

        #@ STOP RUN change2OPT
        # 5. ネットワークグラフの描画

        self.view_nx_matlib4opt()

        #self.view_nx_matlib()


        #@ MOVED
        self.updated_parameters()


        #@ STOP RUN
        # 6. PSIの表示
        if self.root_node_out_opt == None:
            self.root.after(1000, self.show_psi("outbound", "supply"))

            #@ STOP
            ## パラメータの初期化と更新を呼び出し
            #self.updated_parameters()

        else:  # is root_node_out_opt
            self.root.after(1000, self.show_psi_graph4opt)


            #@ STOP
            ## パラメータの初期化と更新を呼び出し
            #self.set_market_potential(self.root_node_out_opt)
            #self.updated_parameters()
            ##self.initialize_parameters()

        # 7. 完了メッセージの表示
        messagebox.showinfo("Load Completed", "Plan data load is completed")




    def on_exit(self):
        # 確認ダイアログの表示
        if messagebox.askokcancel("Quit", "Do you really want to exit?"):
            # 全てのスレッドを終了
            for thread in threading.enumerate():
                if thread is not threading.main_thread():
                    thread.join(timeout=1)
    
            #for widget in self.root.winfo_children():
            #    widget.destroy()
    
            #self.root.destroy()
            self.root.quit()


    # **********************************
    # sub menus
    # **********************************

    #def show_cost_stracture_bar_graph(self):
    #    pass

    # viewing Cost Stracture / an image of Value Chain
    def show_cost_stracture_bar_graph(self):
        try:
            if self.root_node_outbound is None or self.root_node_inbound is None:
                raise ValueError("Data has not been loaded yet")
            
            self.show_nodes_cs_lot_G_Sales_Procure(self.root_node_outbound, self.root_node_inbound)
        
        except ValueError as ve:
            print(f"error: {ve}")
            tk.messagebox.showerror("error", str(ve))
        
        except AttributeError:
            print("Error: Required attributes are missing from the node. Please check if the data is loaded.")
            tk.messagebox.showerror("Error", "Required attributes are missing from the node. Please check if the data is loaded.")
        
        except Exception as e:
            print(f"An unexpected error has occurred: {e}")
            tk.messagebox.showerror("Error", f"An unexpected error has occurred: {e}")




    def show_nodes_cs_lot_G_Sales_Procure(self, root_node_outbound, root_node_inbound):
        attributes = [
            'cs_direct_materials_costs',
            'cs_marketing_promotion',
            'cs_sales_admin_cost',
            'cs_tax_portion',
            'cs_logistics_costs',
            'cs_warehouse_cost',
            'cs_prod_indirect_labor',
            'cs_prod_indirect_others',
            'cs_direct_labor_costs',
            'cs_depreciation_others',
            'cs_profit',
        ]

        def dump_node_amt_all_in(node, node_amt_all):
            for child in node.children:
                dump_node_amt_all_in(child, node_amt_all)
            amt_list = {attr: getattr(node, attr) for attr in attributes}
            if node.name == "JPN":
                node_amt_all["JPN_IN"] = amt_list
            else:
                node_amt_all[node.name] = amt_list
            return node_amt_all

        def dump_node_amt_all_out(node, node_amt_all):
            amt_list = {attr: getattr(node, attr) for attr in attributes}
            if node.name == "JPN":
                node_amt_all["JPN_OUT"] = amt_list
            else:
                node_amt_all[node.name] = amt_list
            for child in node.children:
                dump_node_amt_all_out(child, node_amt_all)
            return node_amt_all

        node_amt_sum_in = dump_node_amt_all_in(root_node_inbound, {})
        node_amt_sum_out = dump_node_amt_all_out(root_node_outbound, {})
        node_amt_sum_in_out = {**node_amt_sum_in, **node_amt_sum_out}

        print("node_amt_sum_out", node_amt_sum_out)

        make_stack_bar4cost_stracture(node_amt_sum_out)

        # CSVファイルへのエクスポートを呼び出す
        self.export_cost_structure_to_csv(root_node_outbound, root_node_inbound, "cost_structure.csv")





    def export_cost_structure_to_csv(self, root_node_outbound, root_node_inbound, file_path):
        attributes = [
            'cs_direct_materials_costs',
            'cs_marketing_promotion',
            'cs_sales_admin_cost',
            'cs_tax_portion',
            'cs_logistics_costs',
            'cs_warehouse_cost',
            'cs_prod_indirect_labor',
            'cs_prod_indirect_others',
            'cs_direct_labor_costs',
            'cs_depreciation_others',
            'cs_profit',
        ]

        def dump_node_amt_all_in(node, node_amt_all):
            for child in node.children:
                dump_node_amt_all_in(child, node_amt_all)
            amt_list = {attr: getattr(node, attr) for attr in attributes}
            if node.name == "JPN":
                node_amt_all["JPN_IN"] = amt_list
            else:
                node_amt_all[node.name] = amt_list
            return node_amt_all

        def dump_node_amt_all_out(node, node_amt_all):
            amt_list = {attr: getattr(node, attr) for attr in attributes}
            if node.name == "JPN":
                node_amt_all["JPN_OUT"] = amt_list
            else:
                node_amt_all[node.name] = amt_list
            for child in node.children:
                dump_node_amt_all_out(child, node_amt_all)
            return node_amt_all

        node_amt_sum_in = dump_node_amt_all_in(root_node_inbound, {})
        node_amt_sum_out = dump_node_amt_all_out(root_node_outbound, {})
        node_amt_sum_in_out = {**node_amt_sum_in, **node_amt_sum_out}

        # 横持ちでデータフレームを作成
        data = []
        for node_name, costs in node_amt_sum_in_out.items():
            row = [node_name] + [costs[attr] for attr in attributes]
            data.append(row)

        df = pd.DataFrame(data, columns=["node_name"] + attributes)

        # CSVファイルにエクスポート
        df.to_csv(file_path, index=False)
        print(f"Cost structure exported to {file_path}")






    def show_month_data_csv(self):
        pass
    
    def outbound_psi_to_csv(self):
        pass
    



    #def outbound_lot_by_lot_to_csv(self):
    #    pass
    def outbound_lot_by_lot_to_csv(self):
        # ファイル保存ダイアログを表示して保存先ファイルパスを取得
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return  # ユーザーがキャンセルした場合

        # 計画の出力期間を計算
        #output_period_outbound = 53 * self.plan_range
        output_period_outbound = 53 * self.root_node_outbound.plan_range

        start_year = self.plan_year_st

        # ヘッダーの作成
        headers = ["tier", "node_name", "parent", "PSI_attribute", "year", "week_no", "lot_id"]

        # データの収集
        data = []

        def collect_data(node, output_period, tier_no, parent_name):
            for attr in range(4):  # 0:"Sales", 1:"CarryOver", 2:"Inventory", 3:"Purchase"
                for week_no in range(output_period):
                    year = start_year + week_no // 53
                    week = week_no % 53 + 1
                    lot_ids = node.psi4supply[week_no][attr]
                    if not lot_ids:  # 空リストの場合、空文字を追加
                        lot_ids = [""]
                    for lot_id in lot_ids:
                        data.append([tier_no, node.name, parent_name, attr, year, week, lot_id])
            for child in node.children:
                collect_data(child, output_period, tier_no + 1, node.name)

        # root_node_outboundのツリー構造を走査してデータを収集
        collect_data(self.root_node_outbound, output_period_outbound, 0, "root")

        # データフレームを作成してCSVファイルに保存
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(save_path, index=False)

        # 完了メッセージを表示
        messagebox.showinfo("CSV Export", f"Lot by Lot data has been exported to {save_path}")
    



    def inbound_psi_to_csv(self):
        pass
    
    def inbound_lot_by_lot_to_csv(self):
        pass
    
    def lot_cost_structure_to_csv(self):
        pass
    
    def supplychain_performance_to_csv(self):
        pass
    
    def psi_for_excel(self):
        pass
    



    #def show_3d_overview(self):
    #    pass

    def show_3d_overview(self):
        # CSVファイルを読み込む
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return  # ユーザーがキャンセルした場合

        # CSVファイルを読み込む
        df = pd.read_csv(file_path)

        # TreeViewを作成してノードを選択させる
        tree_window = tk.Toplevel(self.root)
        tree_window.title("Select Node")
        tree = ttk.Treeview(tree_window)
        tree.pack(fill=tk.BOTH, expand=True)

        # ユニークなノード名のリストを抽出
        node_list = df[['tier', 'node_name', 'parent']].drop_duplicates().sort_values(by='tier')

        # ルートノードを追加
        root_node = tree.insert('', 'end', text='root', iid='root')
        node_id_map = {"root": root_node}

        # ノードをツリー構造に追加
        def add_node(parent, tier, node_name, node_id):
            tree.insert(parent, 'end', node_id, text=f"Tier {tier}: {node_name}")

        for _, row in node_list.iterrows():
            node_id = f"{row['tier']}_{row['node_name']}"
            parent_node_name = row.get("parent", "root")
            if parent_node_name in node_id_map:
                parent = node_id_map[parent_node_name]
                add_node(parent, row["tier"], row["node_name"], node_id)
                node_id_map[row["node_name"]] = node_id
            else:
                # 親ノードが見つからない場合はルートノードを使用
                add_node(root_node, row["tier"], row["node_name"], node_id)
                node_id_map[row["node_name"]] = node_id

        # 選択ボタンの設定
        def select_node():
            selected_item = tree.selection()
            if selected_item:
                node_name = tree.item(selected_item[0], "text").split(": ")[1]
                tree_window.destroy()
                self.plot_3d_graph(df, node_name)

        select_button = tk.Button(tree_window, text="Select", command=select_node)
        select_button.pack()





    def plot_3d_graph(self, df, node_name):
        psi_attr_map = {0: "lightblue", 1: "darkblue", 2: "brown", 3: "gold"}

        x = []
        y = []
        z = []
        labels = []
        colors = []
        week_no_dict = {}
        max_z_value_lot_id_map = {}

        lot_position_map = {}

        for _, row in df.iterrows():
            if row["node_name"] == node_name and pd.notna(row["lot_id"]):
                x_value = row["PSI_attribute"]
                year = row['year']
                week_no = row['week_no']

                # Calculate week_no_serial
                start_year = df['year'].min()
                week_no_serial = (year - start_year) * 53 + week_no
                week_no_dict[week_no_serial] = f"{year}{str(week_no).zfill(2)}"

                y_value = week_no_serial
                lot_id = row['lot_id']

                if (x_value, y_value) not in lot_position_map:
                    lot_position_map[(x_value, y_value)] = 0

                z_value = lot_position_map[(x_value, y_value)] + 1
                lot_position_map[(x_value, y_value)] = z_value

                # Update max z_value for the corresponding (x_value, y_value)
                if (x_value, y_value) not in max_z_value_lot_id_map or z_value > max_z_value_lot_id_map[(x_value, y_value)][0]:
                    max_z_value_lot_id_map[(x_value, y_value)] = (z_value, lot_id)

                x.append(x_value)
                y.append(y_value)
                z.append(z_value)
                labels.append(lot_id)
                colors.append(psi_attr_map[row["PSI_attribute"]])

        # Tkinterのウィンドウを作成
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"3D Plot for {node_name}")

        # Figureを作成
        fig = plt.figure(figsize=(16, 12))  # 図のサイズを指定
        ax = fig.add_subplot(111, projection='3d')

        # 3Dプロットの作成
        scatter = ax.scatter(x, y, z, c=colors, s=1, depthshade=True)  # s=1でプロットサイズを小さく設定
        ax.set_xlabel('PSI Attribute')
        ax.set_ylabel('Time (YYYYWW)')
        ax.set_zlabel('Lot ID Position')

        # x軸のラベル設定
        ax.set_xticks(list(psi_attr_map.keys()))
        ax.set_xticklabels(["Sales", "CarryOver", "Inventory", "Purchase"], rotation=45, ha='right')

        # y軸のラベル設定
        y_ticks = [week_no_serial for week_no_serial in week_no_dict.keys() if week_no_serial % 2 != 0]
        y_labels = [week_no_dict[week_no_serial] for week_no_serial in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, rotation=45, ha='right', fontsize=6)  # フォントサイズをさらに小さく設定

        # 各座標に対応するlot_idの表示（z軸の最大値のみ）
        for (x_value, y_value), (z_value, lot_id) in max_z_value_lot_id_map.items():
            ax.text(x_value, y_value, z_value, lot_id, fontsize=4, color='black', ha='center', va='center')

        # FigureをTkinterのCanvasに追加
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Tkinterのメインループを開始
        plot_window.mainloop()

        # プロットをPNGとして保存
        plt.savefig("interactive_plot.png")
        print("Interactive plot saved as interactive_plot.png")



    
    
    # ******************************
    # define planning ENGINE
    # ******************************

    #def demand_planning(self):
    #    pass

    def demand_planning(self):
        # Implement forward planning logic here
        print("Forward planning executed.")

        #@240903@241106
        calc_all_psi2i4demand(self.root_node_outbound)


        self.update_evaluation_results()

        #@241212 add
        self.decouple_node_selected = []
        self.view_nx_matlib()

        self.root.after(1000, self.show_psi("outbound", "demand"))
        #self.root.after(1000, self.show_psi_graph)
        #self.show_psi_graph() # this event do not live 






    #def demand_leveling(self):
    #    pass

    #@250120 STOP with "name chaged"
    def demand_leveling(self):
        # Demand Leveling logic here
        print("Demand Leveling executed.")


        # *********************************
        # Demand LEVELing on shipping yard / with pre_production week
        # *********************************

        year_st  = 2020
        year_end = 2021

        year_st  = self.plan_year_st
        year_end = year_st + self.plan_range - 1

        pre_prod_week = self.pre_proc_LT

        # STOP
        #year_st = df_capa_year["year"].min()
        #year_end = df_capa_year["year"].max()

        # root_node_outboundのsupplyの"S"のみを平準化して生成している
        demand_leveling_on_ship(self.root_node_outbound, pre_prod_week, year_st, year_end)


        # root_node_outboundのsupplyの"PSI"を生成している
        ##@241114 KEY CODE
        self.root_node_outbound.calcS2P_4supply()  #mother plantのconfirm S=> P
        self.root_node_outbound.calcPS2I4supply()  #mother plantのPS=>I


        #@241114 KEY CODE
        # ***************************************
        # その3　都度のparent searchを実行 setPS_on_ship2node
        # ***************************************
        feedback_psi_lists(self.root_node_outbound, self.nodes_outbound)


        #feedback_psi_lists(self.root_node_outbound, node_psi_dict_Ot4Sp, self.nodes_outbound)


        # STOP
        #decouple_node_names = [] # initial PUSH with NO decouple node
        ##push_pull_on_decouple
        #push_pull_all_psi2i_decouple4supply5(
        #    self.root_node_outbound,
        #    decouple_node_names )



        #@241114 KEY CODE
        #@240903

        #calc_all_psi2i4demand(self.root_node_outbound)
        #calc_all_psi2i4supply(self.root_node_outbound)


        self.update_evaluation_results()

        # PSI計画の初期状態をバックアップ
        self.psi_backup_to_file(self.root_node_outbound, 'psi_backup.pkl')

        self.view_nx_matlib()

        self.root.after(1000, self.show_psi("outbound", "supply"))
        #self.root.after(1000, self.show_psi_graph)




    def psi_backup(self, node, status_name):
        return copy.deepcopy(node)

    def psi_restore(self, node_backup, status_name):
        return copy.deepcopy(node_backup)

    def psi_backup_to_file(self, node, filename):
        with open(filename, 'wb') as file:
            pickle.dump(node, file)

    def psi_restore_from_file(self, filename):
        with open(filename, 'rb') as file:
            node_backup = pickle.load(file)
        return node_backup




    def supply_planning(self):
        # Check if the necessary data is loaded
        if self.root_node_outbound is None or self.nodes_outbound is None:
            print("Error: PSI Plan data is not loaded. Please load the data first.")
            tk.messagebox.showerror("Error", "PSI Plan data is NOT loaded. please File Open parameter directory first.")
            return
    
        # Implement forward planning logic here
        print("Supply planning with Decoupling points")
    
        # Restore PSI data from a backup file
        self.root_node_outbound = self.psi_restore_from_file('psi_backup.pkl')
    
        if self.decouple_node_selected == []:
            # Search nodes_decouple_all[-2], that is "DAD" nodes
            nodes_decouple_all = make_nodes_decouple_all(self.root_node_outbound    )
            print("nodes_decouple_all", nodes_decouple_all)
    
            # [-2] will be "DAD" node, the point of Delivery and Distribution
            decouple_node_names = nodes_decouple_all[-2]
        else:
            decouple_node_names = self.decouple_node_selected
    
        # Perform supply planning logic
        push_pull_all_psi2i_decouple4supply5(
            self.root_node_outbound, decouple_node_names
        )
    
        # Evaluate the results
        self.update_evaluation_results()
    
        # Update the network visualization
        self.decouple_node_selected = decouple_node_names
        self.view_nx_matlib4opt()
    
        # Update the PSI area
        self.root.after(1000, self.show_psi("outbound", "supply"))
    
    
    
    
    #def eval_buffer_stock(self):
    #    pass

    def eval_buffer_stock(self):

        # Check if the necessary data is loaded
        if self.root_node_outbound is None or self.nodes_outbound is None:
            print("Error: PSI Plan data is not loaded. Please load the data first.")
            tk.messagebox.showerror("Error", "PSI Plan data is NOT loaded. please File Open parameter directory first.")
            return

        print("eval_buffer_stock with Decoupling points")

        # This backup is in "demand leveling"
        ## PSI計画の初期状態をバックアップ
        #self.psi_backup_to_file(self.root_node_outbound, 'psi_backup.pkl')

        nodes_decouple_all = make_nodes_decouple_all(self.root_node_outbound)
        print("nodes_decouple_all", nodes_decouple_all)

        for i, decouple_node_names in enumerate(nodes_decouple_all):
            print("nodes_decouple_all", nodes_decouple_all)


            # PSI計画の状態をリストア
            self.root_node_outbound = self.psi_restore_from_file('psi_backup.pkl')

            push_pull_all_psi2i_decouple4supply5(self.root_node_outbound, decouple_node_names)
            self.update_evaluation_results()

            print("decouple_node_names", decouple_node_names)
            print("self.total_revenue", self.total_revenue)
            print("self.total_profit", self.total_profit)

            self.decouple_node_dic[i] = [self.total_revenue, self.total_profit, decouple_node_names]

            ## network area
            #self.view_nx_matlib()

            ##@241207 TEST
            #self.root.after(1000, self.show_psi("outbound", "supply"))


        self.display_decoupling_patterns()
        # PSI area => move to selected_node in window





    def optimize_network(self):

        # Check if the necessary data is loaded
        if self.root_node_outbound is None or self.nodes_outbound is None:
            print("Error: PSI Plan data is not loaded. Please load the data first.")
            tk.messagebox.showerror("Error", "PSI Plan data is NOT loaded. please File Open parameter directory first.")
            return

        print("optimizing start")


    #@ STOP
    #def optimize_and_view_nx_matlib(self):

        G = nx.DiGraph()    # base display field


        Gdm_structure = nx.DiGraph()  # optimise for demand side
        #Gdm = nx.DiGraph()  # optimise for demand side

        Gsp = nx.DiGraph()  # optimise for supply side

        self.G = G
        self.Gdm_structure = Gdm_structure
        self.Gsp = Gsp


        root_node_outbound = self.root_node_outbound 
        nodes_outbound = self.nodes_outbound     
        root_node_inbound = self.root_node_inbound  
        nodes_inbound = self.nodes_inbound      

        pos_E2E, G, Gdm_structure, Gsp = self.show_network_E2E_matplotlib(
        #pos_E2E, Gdm_structure, Gsp = show_network_E2E_matplotlib(
        #pos_E2E, flowDict_dm, flowDict_sp, Gdm_structure, Gsp = show_network_E2E_matplotlib(
            root_node_outbound, nodes_outbound, 
            root_node_inbound, nodes_inbound, 
            G, Gdm_structure, Gsp
        )


        # **************************************************
        # optimizing here
        # **************************************************

        G_opt = Gdm_structure.copy()


        # 最適化パラメータをリセット
        self.reset_optimization_params(G_opt)

        #@241229 ADD
        self.reset_optimized_path(G_opt)

        # 新しい最適化パラメータを設定
        self.set_optimization_params(G_opt)

        flowDict_opt = self.flowDict_opt
        print("optimizing here flowDict_opt", flowDict_opt)


        # 最適化を実行
        # fllowing set should be done here
        #self.flowDict_opt = flowDict_opt
        #self.flowCost_opt = flowCost_opt

        self.run_optimization(G_opt)
        print("1st run_optimization self.flowDict_opt", self.flowDict_opt)







        # flowCost_opt = self.flowCost_opt # direct input

        G_result = G_opt.copy()


        G_view = G_result.copy()
        self.add_optimized_path(G_view, self.flowDict_opt)






        #@241205 STOP **** flowDict_optを使ったGのE2Eの表示系に任せる
        ## 前回の最適化pathをリセット
        self.reset_optimized_path(G_result)
        #
        ## 新しい最適化pathを追加
        G_result = G_opt.copy()
        self.add_optimized_path(G_result, self.flowDict_opt)
        
        # 最適化pathの表示（オプション）
        #print("Iteration", i + 1)
        print("Optimized Path:", self.flowDict_opt)
        print("Optimized Cost:", self.flowCost_opt)


        # make optimized tree and PSI planning and show it
        flowDict_opt = self.flowDict_opt


        optimized_nodes = {} # 初期化
        optimized_nodes = self.create_optimized_tree(flowDict_opt)


        if not optimized_nodes:
            error_message = "error: optimization with NOT enough supply"
            print(error_message)
            self.show_error_message(error_message)  # 画面にエラーメッセージを表示する関数
            return


        print("optimized_nodes", optimized_nodes)
        optimized_root = optimized_nodes['supply_point']
        self.optimized_root = optimized_root



        #@241227 MEMO 
        # 最適化されたnodeの有無でPSI表示をON/OFFしているが、これに加えて
        # ここでは、最適化nodeは存在し、、年間の値が0の時、
        # 年間供給量を月次に按分して供給するなどの処理を追加する



        # *********************************
        # making limited_supply_nodes
        # *********************************
        leaf_nodes_out       = self.leaf_nodes_out  # all leaf_nodes
        optimized_nodes_list = []              # leaf_node on targetted market
        limited_supply_nodes = []              # leaf_node Removed from target

        # 1. optimized_nodes辞書からキー項目をリストoptimized_nodes_listに抽出
        optimized_nodes_list = list(optimized_nodes.keys())

        # 2. leaf_nodes_outからoptimized_nodes_listの要素を排除して
        # limited_supply_nodesを生成
        limited_supply_nodes = [node for node in leaf_nodes_out if node not in optimized_nodes_list]

        # 結果を表示
        print("optimized_nodes_list:", optimized_nodes_list)
        print("limited_supply_nodes:", limited_supply_nodes)


# 最適化の結果をPSIに反映する方法
# 1. 入力ファイルS_month_data.csvをdataframeに読込み
# 2. limited_supply_nodesの各要素node nameに該当するS_month_dataのSの値を
#    すべて0 clearする。
# 3. 結果を"S_month_optimized.csv"として保存する
# 4. S_month_optimized.csvを入力として、load_data_opt_filesからPSI planする



        # limited_supply_nodesのリスト
        #limited_supply_nodes = ['MUC_N', 'MUC_D', 'MUC_I', 'SHA_I', 'NYC_D', 'NYC_I', 'LAX_D', 'LAX_I']












        # 入力CSVファイル名
        input_csv = 'S_month_data.csv'


        # デバッグ用コード追加
        print(f"self.directory: {self.directory}")
        print(f"input_csv: {input_csv}")

        if self.directory is None or input_csv is None:
            raise ValueError("self.directory または input_csv が None になっています。適切な値を設定してください。")


        input_csv_path = os.path.join(self.directory, input_csv)


        # 出力CSVファイル名
        output_csv = 'S_month_optimized.csv'
        output_csv_path = os.path.join(self.directory, output_csv)




        # S_month.csvにoptimized_demandをセットする
        # optimized leaf_node以外を0 clearする


        #@ STOP
        # 最適化にもとづく供給配分 ここでは簡易的にon-offしているのみ
        # 本来であれば、最適化の供給配分を詳細に行うべき所
        #self.clear_s_values(limited_supply_nodes, input_csv_path, output_csv_path)


        input_csv = 'S_month_data.csv'
        output_csv = 'S_month_optimized.csv'

        input_csv_path = os.path.join(self.directory, input_csv)
        output_csv_path = os.path.join(self.directory, output_csv)

        self.clear_s_values(self.flowDict_opt, input_csv_path, output_csv_path)



        ## **************************************
        ## input_csv = 'S_month_optimized.csv' load_files & planning
        ## **************************************
        #
        self.load_data_files4opt()     # loading with 'S_month_optimized.csv'

        #
        self.plan_through_engines4opt()




        # **************************************
        # いままでの評価と描画系
        # **************************************




        # *********************
        # evaluation@241220
        # *********************
        #@241225 memo "root_node_out_opt"のtreeにはcs_xxxxがセットされていない
        self.update_evaluation_results4optimize()


        # *********************
        # network graph
        # *********************
        # STAY ORIGINAL PLAN
        # selfのhandle nameは、root_node_outboundで、root_node_out_optではない
        # 
        # グラフ描画関数を呼び出し  最適ルートを赤線で表示
        #
        # title revenue, profit, profit_ratio
        self.draw_network4opt(G, Gdm_structure, Gsp, pos_E2E, self.flowDict_opt)

        #self.draw_network4opt(G, Gdm, Gsp, pos_E2E, flowDict_dm, flowDict_sp, flowDict_opt)


        # *********************
        # PSI graph
        # *********************
        self.root.after(1000, self.show_psi_graph4opt)
        #self.root.after(1000, self.show_psi_graph)

        #@ ADD
        # パラメータの初期化と更新を呼び出し
        self.updated_parameters()
        #@ STOP
        #self.updated_parameters4opt()









# **** 19 call_backs END*****



# **** Start of SUB_MODULE for Optimization ****

    def _load_tree_structure(self, load_directory):

        with open(f"{load_directory}/root_node_outbound.pkl", 'rb') as f:
            self.root_node_outbound = pickle.load(f)
            print(f"root_node_outbound loaded: {self.root_node_outbound}")

        with open(f"{load_directory}/root_node_inbound.pkl", 'rb') as f:
            self.root_node_inbound = pickle.load(f)
            print(f"root_node_inbound loaded: {self.root_node_inbound}")


        if os.path.exists(f"{load_directory}/root_node_out_opt.pkl"):
            with open(f"{load_directory}/root_node_out_opt.pkl", 'rb') as f:
                self.root_node_out_opt = pickle.load(f)
                print(f"root_node_out_opt loaded: {self.root_node_out_opt}")
        else:
            self.flowDict_opt = {}  # NO optimize
            pass





    def reset_optimization_params(self, G):
        for u, v in G.edges():
            G[u][v]['capacity'] = 0
            G[u][v]['weight'] = 0
        for node in G.nodes():
            G.nodes[node]['demand'] = 0
    
    
    def reset_optimized_path(self, G):
        for u, v in G.edges():
            if 'flow' in G[u][v]:
                del G[u][v]['flow']
    
    
    def run_optimization(self, G):

        #flow_dict = nx.min_cost_flow(G)
        #cost = nx.cost_of_flow(G, flow_dict)
        #return flow_dict, cost


        # ************************************
        # optimize network
        # ************************************
        try:
    
            flowCost_opt, flowDict_opt = nx.network_simplex(G)

        except Exception as e:
            print("Error during optimization:", e)
            return

        self.flowCost_opt = flowCost_opt
        self.flowDict_opt = flowDict_opt

        print("flowDict_opt", flowDict_opt)
        print("flowCost_opt", flowCost_opt)

        print("end optimization")
    
    
    def add_optimized_path(self, G, flow_dict):
        for u in flow_dict:
            for v, flow in flow_dict[u].items():
                if flow > 0:
                    G[u][v]['flow'] = flow
    
    

    # 画面にエラーメッセージを表示する関数
    def show_error_message(self, message):
        error_window = tk.Toplevel(self.root)
        error_window.title("Error")
        tk.Label(error_window, text=message, fg="red").pack()
        tk.Button(error_window, text="OK", command=error_window.destroy).pack()




    # オリジナルノードからコピーする処理
    def copy_node(self, node_name):
        original_node = self.nodes_outbound[node_name]  #オリジナルノードを取得
        copied_node = copy.deepcopy(original_node)  # deepcopyを使ってコピー
        return copied_node




    def create_optimized_tree(self, flowDict_opt):
        # Optimized Treeの生成
        optimized_nodes = {}
        for from_node, flows in flowDict_opt.items():

            if from_node == 'sales_office': # 末端の'sales_office'はtreeの外
                pass
            else:

                for to_node, flow in flows.items():

                    if to_node == 'sales_office': # 末端の'sales_office'はtreeの外
                        pass
                    else:


                        if flow > 0:
                            if from_node not in optimized_nodes:
                                optimized_nodes[from_node] = self.copy_node(from_node)
                            if to_node not in optimized_nodes:
                                optimized_nodes[to_node] = self.copy_node(to_node)
                                optimized_nodes[to_node].parent =optimized_nodes[from_node]
        return optimized_nodes




    def set_optimization_params(self, G):
        print("optimization start")

        #Gdm = self.Gdm

        nodes_outbound = self.nodes_outbound
        root_node_outbound = self.root_node_outbound

        print("root_node_outbound.name", root_node_outbound.name)

        # Total Supply Planの取得
        total_supply_plan = int( self.total_supply_plan )

        #total_supply_plan = int(self.tsp_entry.get())


        print("setting capacity")
        max_capacity = 1000000  # 設定可能な最大キャパシティ（適切な値を設定）
        scale_factor_capacity = 1  # キャパシティをスケールするための因子
        scale_factor_demand   = 1  # スケーリング因子

        for edge in G.edges():
            from_node, to_node = edge

            # if node is leaf_node

            #@250103 STOP
            #if from_node in self.leaf_nodes_out and to_node == 'sales_office':

            #@250103 RUN
            if to_node in self.leaf_nodes_out:


                #@250103 RUN
                # ********************************************
                # scale_factor_capacity
                #@241220 TAX100... demand curve... Price_Up and Demand_Down
                # ********************************************
                capacity = int(nodes_outbound[to_node].nx_capacity * scale_factor_capacity)

                #@ STOP
                ## ********************************************
                ## scale_factor_capacity
                ##@241220 TAX100... demand curve... Price_Up and Demand_Down
                ## ********************************************
                #capacity = int(nodes_outbound[from_node].lot_counts_all * scale_factor_capacity)


                G.edges[edge]['capacity'] = capacity
            else:
                G.edges[edge]['capacity'] = max_capacity  # 最大キャパシティを設定
            print("G.edges[edge]['capacity']", edge, G.edges[edge]['capacity'])


        #@250102 MARK
        print("setting weight")
        for edge in G.edges():
            from_node, to_node = edge


            #@ RUN
            G.edges[edge]['weight'] = int(nodes_outbound[from_node].nx_weight)
            print("weight = nx_weight = cs_cost_total+TAX", nodes_outbound[from_node].name, int(nodes_outbound[from_node].nx_weight) )

            #@ STOP
            #G.edges[edge]['weight'] = int(nodes_outbound[from_node].cs_cost_total)
            #print("weight = cs_cost_total", nodes_outbound[from_node].name, int(nodes_outbound[from_node].cs_cost_total) )


        print("setting source and sink")

        # Total Supply Planの取得
        total_supply_plan = int( self.total_supply_plan )

        print("source:supply_point:-total_supply_plan", -total_supply_plan * scale_factor_demand)
        print("sink  :sales_office:total_supply_plan", total_supply_plan * scale_factor_demand)


        # scale = 1
        G.nodes['supply_point']['demand'] = -total_supply_plan * scale_factor_demand
        G.nodes['sales_office']['demand'] = total_supply_plan * scale_factor_demand


        print("optimizing supply chain network")

        for node in G.nodes():
            if node != 'supply_point' and node != 'sales_office':
                G.nodes[node]['demand'] = 0  # 他のノードのデマンドは0に設定



#        # ************************************
#        # optimize network
#        # ************************************
#        try:
#
#            flowCost_opt, flowDict_opt = nx.network_simplex(G)
#
#        except Exception as e:
#            print("Error during optimization:", e)
#            return
#
#        self.flowCost_opt = flowCost_opt
#        self.flowDict_opt = flowDict_opt
#
#        print("flowDict_opt", flowDict_opt)
#        print("flowCost_opt", flowCost_opt)
#
#        print("end optimization")




    def plan_through_engines4opt(self):

    #@RENAME
    # nodes_out_opt     : nodes_out_opt    
    # root_node_out_opt : root_node_out_opt


        print("planning with OPTIMIZED S")


        # Demand planning
        calc_all_psi2i4demand(self.root_node_out_opt)


        # Demand LEVELing on shipping yard / with pre_production week
        year_st = self.plan_year_st
        year_end = year_st + self.plan_range - 1
        pre_prod_week = self.pre_proc_LT
        demand_leveling_on_ship(self.root_node_out_opt, pre_prod_week, year_st, year_end)
        # root_node_out_optのsupplyの"PSI"を生成している
        self.root_node_out_opt.calcS2P_4supply()  #mother plantのconfirm S=> P
        self.root_node_out_opt.calcPS2I4supply()  #mother plantのPS=>I

        # ***************************************
        # その3　都度のparent searchを実行 setPS_on_ship2node
        # ***************************************
        feedback_psi_lists(self.root_node_out_opt, self.nodes_out_opt)



        #@241208 STOP
        ## Supply planning
        #print("Supply planning with Decoupling points")
        #nodes_decouple_all = make_nodes_decouple_all(self.root_node_out_opt)
        #print("nodes_decouple_all", nodes_decouple_all)
        #
        #for i, decouple_node_names in enumerate(nodes_decouple_all):
        #    decouple_flag = "OFF"
        #    if i == 0:

        decouple_node_names = self.decouple_node_selected

        push_pull_all_psi2i_decouple4supply5(self.root_node_out_opt, decouple_node_names)






# **** End of Optimization ****





    def load_tree_structure(self):
        try:
            file_path = filedialog.askopenfilename(title="Select Tree Structure File")
            if not file_path:
                return
            # Placeholder for loading tree structure
            self.tree_structure = nx.DiGraph()
            self.tree_structure.add_edge("Root", "Child1")
            self.tree_structure.add_edge("Root", "Child2")
            messagebox.showinfo("Success", "Tree structure loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load tree structure: {e}")




    def view_nx_matlib4opt(self):
        G = nx.DiGraph()
        Gdm_structure = nx.DiGraph()
        Gsp = nx.DiGraph()

        self.G = G
        self.Gdm_structure = Gdm_structure
        self.Gsp = Gsp

        pos_E2E, G, Gdm, Gsp = self.show_network_E2E_matplotlib(
            self.root_node_outbound, self.nodes_outbound,
            self.root_node_inbound, self.nodes_inbound,
            G, Gdm_structure, Gsp
        )

        self.pos_E2E = pos_E2E

        #self.draw_network4opt(G, Gdm, Gsp, pos_E2E)

        # グラフ描画関数を呼び出し  最適ルートを赤線で表示

        print("load_from_directory self.flowDict_opt", self.flowDict_opt)

        self.draw_network4opt(G, Gdm_structure, Gsp, pos_E2E, self.flowDict_opt)



    def draw_network4opt(self, G, Gdm, Gsp, pos_E2E, flowDict_opt):

        ## 既存の軸をクリア
        #self.ax_network.clear()

    #def draw_network(self, G, Gdm, Gsp, pos_E2E):

        self.ax_network.clear()  # 図をクリア


        print("draw_network4opt: self.total_revenue", self.total_revenue)
        print("draw_network4opt: self.total_profit", self.total_profit)

        # 評価結果の更新
        ttl_revenue = self.total_revenue
        ttl_profit = self.total_profit
        ttl_profit_ratio = (ttl_profit / ttl_revenue) if ttl_revenue != 0 else 0

        # 四捨五入して表示
        total_revenue = round(ttl_revenue)
        total_profit = round(ttl_profit)
        profit_ratio = round(ttl_profit_ratio * 100, 1)  # パーセント表示


        #ax.set_title(f'Node: {node_name} | REVENUE: {revenue:,} | PROFIT: {profit:,} | PROFIT_RATIO: {profit_ratio}%', fontsize=8)


        # タイトルを設定
        self.ax_network.set_title(f'PySI\nOptimized Supply Chain Network\nREVENUE: {total_revenue:,} | PROFIT: {total_profit:,} | PROFIT_RATIO: {profit_ratio}%', fontsize=10)


        print("ax_network.set_title: total_revenue", total_revenue)
        print("ax_network.set_title: total_profit", total_profit)


#".format(total_revenue, total_profit))


        self.ax_network.axis('off')








        # ノードの形状と色を定義
        node_shapes = ['v' if node in self.decouple_node_selected else 'o' for node in G.nodes()]
        node_colors = ['brown' if node in self.decouple_node_selected else 'lightblue' for node in G.nodes()]


        # ノードの描画
        for node, shape, color in zip(G.nodes(), node_shapes, node_colors):

            nx.draw_networkx_nodes(G, pos_E2E, nodelist=[node], node_size=50, node_color=color, node_shape=shape, ax=self.ax_network)


        # エッジの描画
        for edge in G.edges():
            if edge[0] == "procurement_office" or edge[1] == "sales_office":
                edge_color = 'lightgrey'  # "procurement_office"または"sales_office"に接続するエッジはlightgrey
            elif edge in Gdm.edges():
                edge_color = 'blue'  # outbound（Gdm）のエッジは青
            elif edge in Gsp.edges():
                edge_color = 'green'  # inbound（Gsp）のエッジは緑
            else:
                edge_color = 'lightgrey'  # その他はlightgrey

            nx.draw_networkx_edges(G, pos_E2E, edgelist=[edge], edge_color=edge_color, arrows=False, ax=self.ax_network, width=0.5)

        # 最適化pathの赤線表示
        for from_node, flows in flowDict_opt.items():
            for to_node, flow in flows.items():
                if flow > 0:
                    # "G"の上に描画
                    nx.draw_networkx_edges(self.G, self.pos_E2E, edgelist=[(from_node, to_node)], ax=self.ax_network, edge_color='red', arrows=False, width=0.5)

        # ノードラベルの描画
        node_labels = {node: f"{node}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos_E2E, labels=node_labels, font_size=6, ax=self.ax_network)





        # ***************************
        # title and axis
        # ***************************
        #plt.title("Supply Chain Network end2end")

        #@ STOOOOOOOP
        #plt.title("Optimized Supply Chain Network")
        #self.ax_network.axis('off')  # 軸を非表示にする


        # キャンバスを更新
        self.canvas_network.draw()




    def view_nx_matlib(self):
        G = nx.DiGraph()
        Gdm_structure = nx.DiGraph()
        Gsp = nx.DiGraph()

        print(f"view_nx_matlib before show_network_E2E_matplotlib self.decouple_node_selected: {self.decouple_node_selected}")

        pos_E2E, G, Gdm_structure, Gsp = self.show_network_E2E_matplotlib(
            self.root_node_outbound, self.nodes_outbound,
            self.root_node_inbound, self.nodes_inbound,
            G, Gdm_structure, Gsp
        )

        self.pos_E2E = pos_E2E

        print(f"view_nx_matlib after show_network_E2E_matplotlib self.decouple_node_selected: {self.decouple_node_selected}")

        self.G = G
        self.Gdm_structure = Gdm_structure
        self.Gsp = Gsp

        self.draw_network(self.G, self.Gdm_structure, self.Gsp, self.pos_E2E)



    def draw_network(self, G, Gdm, Gsp, pos_E2E):
        self.ax_network.clear()  # 図をクリア

        # 評価結果の更新
        ttl_revenue = self.total_revenue
        ttl_profit = self.total_profit
        ttl_profit_ratio = (ttl_profit / ttl_revenue) if ttl_revenue != 0 else 0

        # 四捨五入して表示
        total_revenue = round(ttl_revenue)
        total_profit = round(ttl_profit)
        profit_ratio = round(ttl_profit_ratio * 100, 1)  # パーセント表示

        # タイトルを設定
        self.ax_network.set_title(f'PySI\nOptimized Supply Chain Network\nREVENUE: {total_revenue:,} | PROFIT: {total_profit:,} | PROFIT_RATIO: {profit_ratio}%', fontsize=10)

        self.ax_network.axis('off')

        print("draw_network self.decouple_node_selected", self.decouple_node_selected)
        print("draw_network G nodes", list(G.nodes()))
        print("draw_network G edges", list(G.edges()))

        # Node描画
        node_shapes = ['v' if node in self.decouple_node_selected else 'o' for node in G.nodes()]
        node_colors = ['brown' if node in self.decouple_node_selected else 'lightblue' for node in G.nodes()]

        for node, shape, color in zip(G.nodes(), node_shapes, node_colors):
            nx.draw_networkx_nodes(G, pos_E2E, nodelist=[node], node_size=50, node_color=color, node_shape=shape, ax=self.ax_network)

        # Edge描画
        for edge in G.edges():
            edge_color = 'lightgrey' if edge[0] == "procurement_office" or edge[1] == "sales_office" else 'blue' if edge in Gdm.edges() else 'green' if edge in Gsp.edges() else 'gray'
            nx.draw_networkx_edges(G, pos_E2E, edgelist=[edge], edge_color=edge_color, arrows=False, ax=self.ax_network, width=0.5)

        # Labels描画
        node_labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos_E2E, labels=node_labels, font_size=6, ax=self.ax_network)


        #@ STOP
        ## キャンバスの再描画
        #self.canvas_network.draw()

        # キャンバスの再描画
        # 描画処理を待機キューに入れて部分的な描画を実行
        self.canvas_network.draw_idle()





    def display_decoupling_patterns(self):
        subroot = tk.Toplevel(self.root)
        subroot.title("Decoupling Stock Buffer Patterns")

        frame = ttk.Frame(subroot)
        frame.pack(fill='both', expand=True)

        tree = ttk.Treeview(frame, columns=('Revenue', 'Profit', 'Nodes'), show='headings')
        tree.heading('Revenue', text='Revenue')
        tree.heading('Profit', text='Profit')
        tree.heading('Nodes', text='Nodes')
        tree.pack(fill='both', expand=True)

        style = ttk.Style()
        # カラムヘッダを中央揃えにする
        style.configure('Treeview.Heading', anchor='center')

        style.configure('Treeview', rowheight=25)  # 行の高さを設定

        def format_number(value):
            return f"{round(value):,}"

        for i, (revenue, profit, nodes) in self.decouple_node_dic.items():
            formatted_revenue = format_number(revenue)
            formatted_profit = format_number(profit)
            tree.insert('', 'end', values=(formatted_revenue, formatted_profit, ', '.join(nodes)))

        # 列を右寄せに設定する関数
        def adjust_column(tree, col):
            tree.column(col, anchor='e')

        # Revenue と Profit の列を右寄せに設定
        adjust_column(tree, 'Revenue')
        adjust_column(tree, 'Profit')

        selected_pattern = None

        def on_select_pattern(event):
            nonlocal selected_pattern
            item = tree.selection()[0]
            selected_pattern = tree.item(item, 'values')

        tree.bind('<<TreeviewSelect>>', on_select_pattern)

        def on_confirm():
            if selected_pattern:
                self.decouple_node_selected = selected_pattern[2].split(', ')

                print("decouple_node_selected", self.decouple_node_selected)
                self.execute_selected_pattern()

                subroot.destroy()  # サブウィンドウを閉じる

        confirm_button = ttk.Button(subroot, text="SELECT buffering stock", command=on_confirm)
        confirm_button.pack()

        subroot.protocol("WM_DELETE_WINDOW", on_confirm)








    def execute_selected_pattern(self):
        decouple_node_names = self.decouple_node_selected

        # PSI計画の状態をリストア
        self.root_node_outbound = self.psi_restore_from_file('psi_backup.pkl')

        print("exe engine decouple_node_selected", self.decouple_node_selected)

        push_pull_all_psi2i_decouple4supply5(self.root_node_outbound, decouple_node_names)

        self.update_evaluation_results()

        self.view_nx_matlib()
        self.root.after(1000, self.show_psi("outbound", "supply"))




    def load4execute_selected_pattern(self):


        # 1. Load元となるdirectoryの問い合わせ
        load_directory = filedialog.askdirectory()
        if not load_directory:
            return  # ユーザーがキャンセルした場合

        ## 2. 初期処理のcsv fileのコピー
        #for filename in os.listdir(load_directory):
        #    if filename.endswith('.csv'):
        #        full_file_name = os.path.join(load_directory, filename)
        #        if os.path.isfile(full_file_name):
        #            shutil.copy(full_file_name, self.directory)

        # 3. Tree構造の読み込み
        with open(os.path.join(load_directory, 'root_node_outbound.pkl'), 'rb') as f:
            self.root_node_outbound = pickle.load(f)
            print(f"root_node_outbound loaded: {self.root_node_outbound.name}")

        #
        #with open(os.path.join(load_directory, 'root_node_inbound.pkl'), 'rb') as f:
        #    self.root_node_inbound = pickle.load(f)
        #    print(f"root_node_inbound loaded: {self.root_node_inbound}")

        # 4. PSIPlannerAppのデータ・インスタンスの読み込み
        with open(os.path.join(load_directory, 'psi_planner_app.pkl'), 'rb') as f:
            loaded_attributes = pickle.load(f)
            self.__dict__.update(loaded_attributes)
            print(f"loaded_attributes: {loaded_attributes}")

        ## 5. nodes_outboundとnodes_inboundを再生成
        #self.nodes_outbound = self.regenerate_nodes(self.root_node_outbound)
        #self.nodes_inbound = self.regenerate_nodes(self.root_node_inbound)

        # network area
        print("load_from_directory self.decouple_node_selected", self.decouple_node_selected)



        #decouple_node_names = self.decouple_node_selected





        decouple_node_names = self.decouple_node_selected

        ## PSI計画の状態をリストア
        #self.root_node_outbound = self.psi_restore_from_file('psi_backup.pkl')

        print("exe engine decouple_node_selected", self.decouple_node_selected)

        push_pull_all_psi2i_decouple4supply5(self.root_node_outbound, decouple_node_names)

        self.update_evaluation_results()


        #@241212 Gdm_structureにupdated
        self.draw_network(G, Gdm_structure, Gsp, pos_E2E)

        ## 追加: キャンバスを再描画
        #self.canvas_network.draw()
        #
        #self.view_nx_matlib()

        self.root.after(1000, self.show_psi("outbound", "supply"))




# ******************************************
# clear_s_values
# ******************************************
#
#複数年のデータに対応するために、node_name と year をキーにして各ノードのデータを処理。
#
#説明
#leaf_nodeの特定方法の修正：
#
#flow_dict 内で各ノードに sales_office が含まれているかどうかで leaf_nodes を特定します。
#
#rule-1, rule-2, rule-3 の適用：
#
#rule-1: flow_dict に存在しないノードの月次Sの値を0に設定。
#
#rule-2: flow_dict に存在し、sales_office に繋がるノードの値が0である場合、月次S#の値を0に設定。
#
#rule-3: flow_dict に存在し、sales_office に繋がるノードの値が0以外である場合、月次Sの値をプロポーションに応じて分配。
#
#proportionsの計算と値の丸め：
#
#各月のproportionを計算し、それを使って丸めた値を求めます。
#
#rounded_values に丸めた値を格納し、合計が期待する供給量と一致しない場合は、
#最大の値を持つ月で調整します。
#
#年間total_supplyが0の場合の処理：
#年間total_supplyが0の場合は、月次Sの値をすべて0に設定します。


    def clear_s_values(self, flow_dict, input_csv, output_csv):
        # 1. 入力ファイルS_month_data.csvをデータフレームに読み込み
        df = pd.read_csv(input_csv)

        # leaf_nodeを特定
        leaf_nodes = [node for node, connections in flow_dict.items() if 'sales_office' in connections]

        # 2. rule-1, rule-2, rule-3を適用してデータを修正する
        for index, row in df.iterrows():
            node_name = row['node_name']
            year = row['year']
            
            if node_name in flow_dict:
                # ノードがflow_dictに存在する場合
                if node_name in leaf_nodes:
                    # rule-2: ノードの値が0の場合、月次Sの値をすべて0に設定
                    if flow_dict[node_name]['sales_office'] == 0:
                        df.loc[(df['node_name'] == node_name) & (df['year'] == year),
                               ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']] = 0
                    else:
                        # rule-3: ノードの値が0以外の場合、月次Sのproportionに応じて分配
                        total_supply = sum(df.loc[(df['node_name'] == node_name) & (df['year'] == year), 
                                                  ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']].values.flatten())
                        if total_supply != 0:
                            proportions = df.loc[(df['node_name'] == node_name) & (df['year'] == year), 
                                                 ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']].values.flatten() / total_supply
                            rounded_values = [round(proportion * flow_dict[node_name]['sales_office']) for proportion in proportions]
                            difference = flow_dict[node_name]['sales_office'] - sum(rounded_values)
                            if difference != 0:
                                max_index = rounded_values.index(max(rounded_values))
                                rounded_values[max_index] += difference
                            df.loc[(df['node_name'] == node_name) & (df['year'] == year),
                                   ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']] = rounded_values
                        else:
                            # 供給量がゼロの場合、元データを保持（エラーチェック）
                            df.loc[(df['node_name'] == node_name) & (df['year'] == year), 
                                   ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']] = [0] * 12
            else:
                # rule-1: ノードがflow_dictに存在しない場合、月次Sの値をすべて0に設定
                df.loc[index, ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']] = 0

        # 3. 結果を"S_month_data_optimized.csv"として保存する
        df.to_csv(output_csv, index=False)
        print(f"Optimized data saved to {output_csv}")




    def view_nx_matlib4opt_OLD(self):
        try:
            if self.tree_structure is None:
                raise ValueError("Tree structure is not loaded.")

            self.ax_network.clear()
            nx.draw(self.tree_structure, with_labels=True, ax=self.ax_network)
            self.canvas_network.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display optimization graph: {e}")








    def eval_supply_chain_cost4opt(self, node_opt):
    
        # change from "out_opt" to "outbound"
        node = self.nodes_outbound[node_opt.name]

        # *********************
        # counting Purchase Order
        # *********************
        # psi_listのPOは、psi_list[w][3]の中のlot_idのロット数=リスト長

        # lot_counts is "out_opt"side
        node_opt.set_lot_counts()

        #@ STOP
        #node.set_lot_counts()
    
        # output:
        #    self.lot_counts_all = sum(self.lot_counts)

        # change lot_counts from "out_opt"side to "outbound"side
        node.lot_counts_all = node_opt.lot_counts_all
    
        # *********************
        # EvalPlanSIP()の中でnode instanceに以下をセットする
        # self.profit, self.revenue, self.profit_ratio
        # *********************
    
        # by weekの計画状態xxx[w]の変化を評価して、self.eval_xxxにセット
        total_revenue, total_profit = node.EvalPlanSIP_cost()

    
        #@241225 ADD
        node.total_revenue     = total_revenue    
        node.total_profit      = total_profit     
                                 
        node_opt.total_revenue = total_revenue
        node_opt.total_profit  = total_profit 
                                 
        self.total_revenue += total_revenue
        self.total_profit  += total_profit
    
    
        #@241118 "eval_" is 1st def /  "eval_cs_" is 2nd def
        # print(
        #    "Eval node profit revenue profit_ratio",
        #    node.name,
        #    node.eval_profit,
        #    node.eval_revenue,
        #    node.eval_profit_ratio,
        # )
    
        for child in node.children:
    
            self.eval_supply_chain_cost4opt(child)




    def update_evaluation_results(self):


        # Evaluation on PSI
        self.total_revenue = 0
        self.total_profit  = 0
        self.profit_ratio  = 0


        # ***********************
        # This is a simple Evaluation process with "cost table"
        # ***********************


#@241120 STOP
#        self.eval_plan()
#
#    def eval_plan(self):



        # 在庫係数の計算
        # I_cost_coeff = I_total_qty_init / I_total_qty_planned
        #
        # 計画された在庫コストの算定
        # I_cost_planned = I_cost_init * I_cost_coeff
    
    
        # by node evaluation Revenue / Cost / Profit
        # "eval_xxx" = "lot_counts" X "cs_xxx" that is from cost_table
        # Inventory cost has 係数 = I_total on Demand/ I_total on Supply
    
    
        #self.total_revenue = 0
        #self.total_profit  = 0
    
        #eval_supply_chain_cost(self.root_node_outbound)
        #self.eval_supply_chain_cost(self.root_node_outbound)
    
        #eval_supply_chain_cost(self.root_node_inbound)
        #self.eval_supply_chain_cost(self.root_node_inbound)

        #@ CONTEXT グローバル変数 STOP
        ## サプライチェーン全体のコストを評価
        #eval_supply_chain_cost(self.root_node_outbound, self)
        #eval_supply_chain_cost(self.root_node_inbound, self)




        # サプライチェーンの評価を開始

        # tree.py に配置して、node に対して：
        # set_lot_counts() を呼び出し、ロット数を設定
        # EvalPlanSIP_cost() で revenue と profit を計算
        # 子ノード (children) に対して再帰的に eval_supply_chain_cost() をcall

        self.total_revenue, self.total_profit = eval_supply_chain_cost(self.root_node_outbound)



        ttl_revenue = self.total_revenue
        ttl_profit  = self.total_profit

        if ttl_revenue == 0:
            ttl_profit_ratio = 0
        else:
            ttl_profit_ratio = ttl_profit / ttl_revenue

        # 四捨五入して表示 
        total_revenue = round(ttl_revenue) 
        total_profit = round(ttl_profit) 
        profit_ratio = round(ttl_profit_ratio*100, 1) # パーセント表示

        print("total_revenue", total_revenue)
        print("total_profit", total_profit)
        print("profit_ratio", profit_ratio)


#total_revenue 343587
#total_profit 32205
#profit_ratio 9.4


        self.total_revenue_entry.config(state='normal')
        self.total_revenue_entry.delete(0, tk.END)
        self.total_revenue_entry.insert(0, f"{total_revenue:,}")
        #self.total_revenue_entry.insert(0, str(kpi_results["total_revenue"]))
        self.total_revenue_entry.config(state='readonly')


        self.total_profit_entry.config(state='normal')
        self.total_profit_entry.delete(0, tk.END)
        self.total_profit_entry.insert(0, f"{total_profit:,}")
        #self.total_profit_entry.insert(0, str(kpi_results["total_profit"]))
        self.total_profit_entry.config(state='readonly')


        self.profit_ratio_entry.config(state='normal')
        self.profit_ratio_entry.delete(0, tk.END)
        self.profit_ratio_entry.insert(0, f"{profit_ratio}%")
        self.profit_ratio_entry.config(state='readonly')

        # 画面を再描画
        self.total_revenue_entry.update_idletasks()
        self.total_profit_entry.update_idletasks()
        self.profit_ratio_entry.update_idletasks()





    def update_evaluation_results4optimize(self):


        # Evaluation on PSI
        self.total_revenue = 0
        self.total_profit  = 0
        self.profit_ratio  = 0


        # ***********************
        # This is a simple Evaluation process with "cost table"
        # ***********************


        # 在庫係数の計算
        # I_cost_coeff = I_total_qty_init / I_total_qty_planned
        #
        # 計画された在庫コストの算定
        # I_cost_planned = I_cost_init * I_cost_coeff
    
    
        # by node evaluation Revenue / Cost / Profit
        # "eval_xxx" = "lot_counts" X "cs_xxx" that is from cost_table
        # Inventory cost has 係数 = I_total on Demand/ I_total on Supply
    
    
        #self.total_revenue = 0
        #self.total_profit  = 0
    


        #@241225 memo "root_node_out_opt"のtreeにはcs_xxxxがセットされていない
        # cs_xxxxのあるnode = self.nodes_outbound[node_opt.name]に変換して参照
        #@241225 be checkek
        # ***************************
        # change ROOT HANDLE
        # ***************************
        self.eval_supply_chain_cost4opt(self.root_node_out_opt)

        print("self.root_node_out_opt.name", self.root_node_out_opt.name)

        #self.eval_supply_chain_cost(self.root_node_outbound)
        #self.eval_supply_chain_cost(self.root_node_inbound)

        ttl_revenue = self.total_revenue
        ttl_profit  = self.total_profit

        print("def update_evaluation_results4optimize")
        print("self.total_revenue", self.total_revenue)
        print("self.total_profit" , self.total_profit)


        if ttl_revenue == 0:
            ttl_profit_ratio = 0
        else:
            ttl_profit_ratio = ttl_profit / ttl_revenue

        # 四捨五入して表示 
        total_revenue = round(ttl_revenue) 
        total_profit = round(ttl_profit) 
        profit_ratio = round(ttl_profit_ratio*100, 1) # パーセント表示

        print("total_revenue", total_revenue)
        print("total_profit", total_profit)
        print("profit_ratio", profit_ratio)


#total_revenue 343587
#total_profit 32205
#profit_ratio 9.4


        self.total_revenue_entry.config(state='normal')
        self.total_revenue_entry.delete(0, tk.END)
        self.total_revenue_entry.insert(0, f"{total_revenue:,}")
        #self.total_revenue_entry.insert(0, str(kpi_results["total_revenue"]))
        self.total_revenue_entry.config(state='readonly')


        self.total_profit_entry.config(state='normal')
        self.total_profit_entry.delete(0, tk.END)
        self.total_profit_entry.insert(0, f"{total_profit:,}")
        #self.total_profit_entry.insert(0, str(kpi_results["total_profit"]))
        self.total_profit_entry.config(state='readonly')


        self.profit_ratio_entry.config(state='normal')
        self.profit_ratio_entry.delete(0, tk.END)
        self.profit_ratio_entry.insert(0, f"{profit_ratio}%")
        self.profit_ratio_entry.config(state='readonly')

        # 画面を再描画
        self.total_revenue_entry.update_idletasks()
        self.total_profit_entry.update_idletasks()
        self.profit_ratio_entry.update_idletasks()




# ******************************************
# visualize graph
# ******************************************

    def view_nx_matlib_stop_draw(self):
        G = nx.DiGraph()
        Gdm_structure = nx.DiGraph()
        Gsp = nx.DiGraph()

        print(f"view_nx_matlib before show_network_E2E_matplotlib self.decouple_node_selected: {self.decouple_node_selected}")

        pos_E2E, G, Gdm_structure, Gsp = self.show_network_E2E_matplotlib(
            self.root_node_outbound, self.nodes_outbound,
            self.root_node_inbound, self.nodes_inbound,
            G, Gdm_structure, Gsp
        )

        self.pos_E2E = pos_E2E

        print(f"view_nx_matlib after show_network_E2E_matplotlib self.decouple_node_selected: {self.decouple_node_selected}")

        self.G = G
        self.Gdm_structure = Gdm_structure
        self.Gsp = Gsp

        #@250106 STOP draw
        #self.draw_network(self.G, self.Gdm_structure, self.Gsp, self.pos_E2E)



    def initialize_graphs(self):
        self.G = nx.DiGraph()
        self.Gdm_structure = nx.DiGraph()
        self.Gsp = nx.DiGraph()


    # ***************************
    # make network with NetworkX
    # ***************************



    def show_network_E2E_matplotlib(self,
            root_node_outbound, nodes_outbound, 
            root_node_inbound, nodes_inbound, 
            G, Gdm, Gsp):
        
        # Original code's logic to process and set up the network
        root_node_name_out = root_node_outbound.name 
        root_node_name_in  = root_node_inbound.name

        total_demand =0
        total_demand = set_leaf_demand(root_node_outbound, total_demand)
        #total_demand = self.set_leaf_demand(root_node_outbound, total_demand)
        print("average_total_demand", total_demand)
        print("root_node_outbound.nx_demand", root_node_outbound.nx_demand)

        root_node_outbound.nx_demand = total_demand  
        root_node_inbound.nx_demand = total_demand  

        G_add_nodes_from_tree(root_node_outbound, G)
        #self.G_add_nodes_from_tree(root_node_outbound, G)
        G_add_nodes_from_tree_skip_root(root_node_inbound, root_node_name_in, G)
        #self.G_add_nodes_from_tree_skip_root(root_node_inbound, root_node_name_in, G)

        G.add_node("sales_office", demand=total_demand)
        G.add_node(root_node_outbound.name, demand=0)
        G.add_node("procurement_office", demand=(-1 * total_demand))

        G_add_edge_from_tree(root_node_outbound, G)
        #self.G_add_edge_from_tree(root_node_outbound, G)
        supplyers_capacity = root_node_inbound.nx_demand * 2 
        G_add_edge_from_inbound_tree(root_node_inbound, supplyers_capacity, G)
        #self.G_add_edge_from_inbound_tree(root_node_inbound, supplyers_capacity, G)

        G_add_nodes_from_tree(root_node_outbound, Gdm)
        #self.G_add_nodes_from_tree(root_node_outbound, Gdm)
        Gdm.add_node(root_node_outbound.name, demand = (-1 * total_demand))
        Gdm.add_node("sales_office", demand = total_demand)
        Gdm_add_edge_sc2nx_outbound(root_node_outbound, Gdm)
        #self.Gdm_add_edge_sc2nx_outbound(root_node_outbound, Gdm)

        G_add_nodes_from_tree(root_node_inbound, Gsp)
        #self.G_add_nodes_from_tree(root_node_inbound, Gsp)
        Gsp.add_node("procurement_office", demand = (-1 * total_demand))
        Gsp.add_node(root_node_inbound.name, demand = total_demand)
        Gsp_add_edge_sc2nx_inbound(root_node_inbound, Gsp)
        #self.Gsp_add_edge_sc2nx_inbound(root_node_inbound, Gsp)

        pos_E2E = make_E2E_positions(root_node_outbound, root_node_inbound)
        #pos_E2E = self.make_E2E_positions(root_node_outbound, root_node_inbound)
        pos_E2E = tune_hammock(pos_E2E, nodes_outbound, nodes_inbound)
        #pos_E2E = self.tune_hammock(pos_E2E, nodes_outbound, nodes_inbound)

        return pos_E2E, G, Gdm, Gsp



    def show_network_E2E_matplotlib_with_self(self):
        root_node_outbound = self.root_node_outbound
        nodes_outbound = self.nodes_outbound
        root_node_inbound = self.root_node_inbound
        nodes_inbound = self.nodes_inbound
        return self.show_network_E2E_matplotlib(
            root_node_outbound, nodes_outbound, 
            root_node_inbound, nodes_inbound, 
            self.G, self.Gdm_structure, self.Gsp
        )


# ******************************************
# optimize network graph
# ******************************************


    def optimize(self, G_opt):
        self.reset_optimization_params(G_opt)
        self.set_optimization_params(G_opt)
        self.run_optimization(G_opt)
        print("run_optimization self.flowDict_opt", self.flowDict_opt)
        self.reset_optimized_path(G_opt)
        self.add_optimized_path(G_opt, self.flowDict_opt)
        print("Optimized Path:", self.flowDict_opt)
        print("Optimized Cost:", self.flowCost_opt)





    def load_data_files4opt(self):

    #@RENAME
    # nodes_outbound     : nodes_out_opt    
    # root_node_outbound : root_node_out_opt


        # setting directory from "plan"
        directory = self.directory


        #@ STOP
        #directory = filedialog.askdirectory(title="Select Data Directory")


        if directory:

            # ***********************
            # Lot sizeを取得して変換
            # ***********************
            #try:
            #    self.lot_size = int(self.lot_size_entry.get())
            #except ValueError:
            #    print("Invalid lot size input. Using default value.")

            # Lot size, Plan Year Start, and Plan Rangeを取得して変換
            try:
                self.lot_size = int(self.lot_size_entry.get())
                self.plan_year_st = int(self.plan_year_entry.get())
                self.plan_range = int(self.plan_range_entry.get())
            except ValueError:
                print("Invalid input for lot size, plan year start, or plan range. Using default values.")

            self.outbound_data = []
            self.inbound_data = []

            print("os.listdir(directory)",os.listdir(directory))

            data_file_list = os.listdir(directory)


            # save directory
            self.directory = directory


            # ************************
            # read "profile_tree_outbound.csv"
            # build tree_outbound
            # ************************
            if "profile_tree_outbound.csv" in data_file_list:

                filename = "profile_tree_outbound.csv"

                file_path = os.path.join(directory, filename)
                #filepath = os.path.join(directory, filename)


                #load_outbound(outbound_tree_file)


                # ***************************
                # set file name for "profile tree"
                # ***************************
                #outbound_tree_file = "profile_tree_outbound.csv"
                #inbound_tree_file = "profile_tree_inbound.csv"

                # ***************************
                # create supply chain tree for "out"bound + optimization
                # ***************************

                # because of the python interpreter performance point of view,
                # this "create tree" code be placed in here, main process

            #@240830
            # "nodes_xxxx" is dictionary to get "node pointer" from "node name"
                nodes_out_opt = {}
                nodes_out_opt, root_node_name_out = create_tree_set_attribute(file_path)

                print("root_node_name_out",root_node_name_out)

                root_node_out_opt = nodes_out_opt[root_node_name_out]




                def make_leaf_nodes(node, list):
                    if node.children == []: # leaf_nodeの場合
                        list.append(node.name)
                    else:
                        pass

                    for child in node.children:
                        make_leaf_nodes(child, list)

                    return list

                leaf_nodes_opt = []
                leaf_nodes_opt = make_leaf_nodes(root_node_out_opt, leaf_nodes_opt)



                # making balance for nodes



                # ********************************
                # set outbound tree handle
                # ********************************
                self.nodes_out_opt = nodes_out_opt
                self.root_node_out_opt = root_node_out_opt


                print("leaf_nodes_opt", leaf_nodes_opt)
                self.leaf_nodes_opt = leaf_nodes_opt

                # ********************************
                # tree wideth/depth count and adjust
                # ********************************
                set_positions(root_node_out_opt)



                # root_node_out_opt = nodes_out_opt['JPN']      # for test, direct define
                # root_node_out_opt = nodes_out_opt['JPN_OUT']  # for test, direct define

                # setting parent on its child
                set_parent_all(root_node_out_opt)
                print_parent_all(root_node_out_opt)

            else:
                print("error: profile_tree_outbound.csv is missed")
                pass


            # ************************
            # read "profile_tree_inbound.csv"
            # build tree_inbound
            # ************************
            if "profile_tree_inbound.csv" in data_file_list:

                filename = "profile_tree_inbound.csv"
                file_path = os.path.join(directory, filename)


                # ***************************
                # create supply chain tree for "in"bound
                # ***************************
                nodes_inbound = {}

                nodes_inbound, root_node_name_in = create_tree_set_attribute(file_path)
                root_node_inbound = nodes_inbound[root_node_name_in]


                # ********************************
                # set inbound tree handle
                # ********************************
                self.nodes_inbound = nodes_inbound
                self.root_node_inbound = root_node_inbound


                # ********************************
                # tree wideth/depth count and adjust
                # ********************************
                set_positions(root_node_inbound)

                # setting parent on its child
                set_parent_all(root_node_inbound)
                print_parent_all(root_node_inbound)

            else:
                print("error: profile_tree_inbound.csv is missed")

                pass




            # ************************
            # read "node_cost_table_outbound.csv"
            # read_set_cost
            # ************************
            if "node_cost_table_outbound.csv" in data_file_list:

                filename = "node_cost_table_outbound.csv"
                file_path = os.path.join(directory, filename)

                read_set_cost(file_path, nodes_out_opt)

            else:
                print("error: node_cost_table_outbound.csv is missed")

                pass




            # ************************
            # read "node_cost_table_inbound.csv"
            # read_set_cost
            # ************************
            if "node_cost_table_inbound.csv" in data_file_list:

                filename = "node_cost_table_inbound.csv"
                file_path = os.path.join(directory, filename)

                read_set_cost(file_path, nodes_inbound)


            else:
                print("error: node_cost_table_inbound.csv is missed")

                pass









            # ***************************
            # make price chain table
            # ***************************

            # すべてのパスを見つける
            paths = find_paths(root_node_out_opt)

            # 各リストをタプルに変換してsetに変換し、重複を排除
            unique_paths = list(set(tuple(x) for x in paths))

            # タプルをリストに戻す
            unique_paths = [list(x) for x in unique_paths]

            print("")
            print("")

            for path in unique_paths:
                print(path)

            sorted_paths = sorted(paths, key=len)

            print("")
            print("")

            for path in sorted_paths:
                print(path)


            #@241224 MARK4OPT_SAVE
            # ************************
            # read "S_month_optimized.csv"
            # trans_month2week2lot_id_list
            # ************************
            if "S_month_optimized.csv" in data_file_list:
            #if "S_month_data.csv" in data_file_list:

                filename = "S_month_optimized.csv"
                in_file_path = os.path.join(directory, filename)


                print("self.lot_size",self.lot_size)

                # 使用例
                #in_file = "S_month_data.csv"

                df_weekly, plan_range, plan_year_st = process_monthly_demand(in_file_path, self.lot_size)

                #df_weekly, plan_range, plan_year_st = trans_month2week2lot_id_list(in_file_path, self.lot_size)


                print("plan_year_st",plan_year_st)
                print("plan_range",plan_range)

                # update plan_year_st plan_range
                self.plan_year_st = plan_year_st  # S_monthで更新
                self.plan_range   = plan_range    # S_monthで更新


                # Update the GUI fields
                self.plan_year_entry.delete(0, tk.END)
                self.plan_year_entry.insert(0, str(self.plan_year_st))
                self.plan_range_entry.delete(0, tk.END)
                self.plan_range_entry.insert(0, str(self.plan_range))


                out_file = "S_iso_week_data_opt.csv"
                out_file_path = os.path.join(directory, out_file)

                df_weekly.to_csv(out_file_path, index=False)

                df_capa_year = make_capa_year_month(in_file_path)

                #@241112 test
                year_st = df_capa_year["year"].min()
                year_end = df_capa_year["year"].max()
                print("year_st, year_end",year_st, year_end)

            else:
                print("error: S_month_optimized.csv is missed")

                pass


            #@241124 ココは、初期のEVAL処理用パラメータ。現在は使用していない
            # planning parameterをNode method(=self.)でセットする。
            # plan_range, lot_counts, cash_in, cash_out用のparameterをセット

            root_node_out_opt.set_plan_range_lot_counts(plan_range, plan_year_st)
            root_node_inbound.set_plan_range_lot_counts(plan_range, plan_year_st)

            # ***************************
            # an image of data
            #
            # for node_val in node_yyyyww_value:
            #   #print( node_val )
            #
            ##['SHA_N', 22.580645161290324, 22.580645161290324, 22.580645161290324, 22.5    80645161290324, 26.22914349276974, 28.96551724137931, 28.96551724137931, 28.    96551724137931, 31.067853170189103, 33.87096774193549, 33.87096774193549, 33    .87096774193549, 33.87096774193549, 30.33333333333333, 30.33333333333333, 30    .33333333333333, 30.33333333333333, 31.247311827956988, 31.612903225806452,

            # node_yyyyww_key [['CAN', 'CAN202401', 'CAN202402', 'CAN202403', 'CAN20240    4', 'CAN202405', 'CAN202406', 'CAN202407', 'CAN202408', 'CAN202409', 'CAN202    410', 'CAN202411', 'CAN202412', 'CAN202413', 'CAN202414', 'CAN202415', 'CAN2    02416', 'CAN202417', 'CAN202418', 'CAN202419',

            # ********************************
            # make_node_psi_dict
            # ********************************
            # 1. treeを生成して、nodes[node_name]辞書で、各nodeのinstanceを操作        する
            # 2. 週次S yyyywwの値valueを月次Sから変換、
            #    週次のlotの数Slotとlot_keyを生成、
            # 3. ロット単位=lot_idとするリストSlot_id_listを生成しながらpsi_list        生成
            # 4. node_psi_dict=[node1: psi_list1,,,]を生成、treeのnode.psi4deman        dに接続する
        
            S_week = []
        
            # *************************************************
            # node_psi辞書を初期セットする
            # initialise node_psi_dict
            # *************************************************
            node_psi_dict = {}  # 変数 node_psi辞書
        
            # ***************************
            # outbound psi_dic
            # ***************************
            node_psi_dict_Ot4Dm = {}  # node_psi辞書Outbound4Demand plan
            node_psi_dict_Ot4Sp = {}  # node_psi辞書Outbound4Supply plan
        
            # coupling psi
            node_psi_dict_Ot4Cl = {}  # node_psi辞書Outbound4Couple plan

            # accume psi
            node_psi_dict_Ot4Ac = {}  # node_psi辞書Outbound4Accume plan
        
            # ***************************
            # inbound psi_dic
            # ***************************
            node_psi_dict_In4Dm = {}  # node_psi辞書Inbound4demand plan
            node_psi_dict_In4Sp = {}  # node_psi辞書Inbound4supply plan
        
            # coupling psi
            node_psi_dict_In4Cl = {}  # node_psi辞書Inbound4couple plan
        
            # accume psi
            node_psi_dict_In4Ac = {}  # node_psi辞書Inbound4accume plan

            # ***************************
            # rootからtree nodeをpreorder順に検索 node_psi辞書に空リストをセット        する
            # psi_list = [[[] for j in range(4)] for w in range(53 * plan_range)        ]
            # ***************************
            node_psi_dict_Ot4Dm = make_psi_space_dict(
        root_node_out_opt, node_psi_dict_Ot4Dm, plan_range
            )
            node_psi_dict_Ot4Sp = make_psi_space_dict(
                root_node_out_opt, node_psi_dict_Ot4Sp, plan_range
            )
            node_psi_dict_Ot4Cl = make_psi_space_dict(
                root_node_out_opt, node_psi_dict_Ot4Cl, plan_range
            )
            node_psi_dict_Ot4Ac = make_psi_space_dict(
                root_node_out_opt, node_psi_dict_Ot4Ac, plan_range
            )
        
            node_psi_dict_In4Dm = make_psi_space_dict(
                root_node_inbound, node_psi_dict_In4Dm, plan_range
            )
            node_psi_dict_In4Sp = make_psi_space_dict(
                root_node_inbound, node_psi_dict_In4Sp, plan_range
            )
            node_psi_dict_In4Cl = make_psi_space_dict(
                root_node_inbound, node_psi_dict_In4Cl, plan_range
            )
            node_psi_dict_In4Ac = make_psi_space_dict(
                root_node_inbound, node_psi_dict_In4Ac, plan_range
            )
        
            # ***********************************
            # set_dict2tree
            # ***********************************
            # rootからtreeをpreorder順に検索
            # node_psi辞書内のpsi_list pointerをNodeのnode objectにsetattr()で接        続
        
            set_dict2tree_psi(root_node_out_opt, "psi4demand", node_psi_dict_Ot4Dm)
            set_dict2tree_psi(root_node_out_opt, "psi4supply", node_psi_dict_Ot4Sp)
            set_dict2tree_psi(root_node_out_opt, "psi4couple", node_psi_dict_Ot4Cl)
            set_dict2tree_psi(root_node_out_opt, "psi4accume", node_psi_dict_Ot4Ac)
        
            set_dict2tree_psi(root_node_inbound, "psi4demand", node_psi_dict_In4Dm)
            set_dict2tree_psi(root_node_inbound, "psi4supply", node_psi_dict_In4Sp)
            set_dict2tree_psi(root_node_inbound, "psi4couple", node_psi_dict_In4Cl)
            set_dict2tree_psi(root_node_inbound, "psi4accume", node_psi_dict_In4Ac)
        








            #@241224 MARK4OPT_SAVE
            #
            # ココで、root_node_out_optのPSIがsetされ、planning engineに渡る
            #
            # ************************************
            # setting S on PSI
            # ************************************

            # Weekly Lot: CPU:Common Planning UnitをPSI spaceにセットする
            set_df_Slots2psi4demand(root_node_out_opt, df_weekly)



            #@241124 adding for "global market potential"
            # ************************************
            # counting all lots
            # ************************************

            #print("check lots on psi4demand[w][0] ")

            ## count lot on all nodes  from  node.psi4demand[w][0] 
            #lot_num = count_lot_all_nodes(root_node_out_opt)

            # year_st
            # year_end

            # **************************************
            # count_lots_on_S_psi4demand
            # **************************************
            # psi4demand[w][0]の配置されたSのlots数を年別にcountしてlist化


            def count_lots_on_S_psi4demand(node, S_list):
                if node.children == []:
                    for w_psi in node.psi4demand:
                        S_list.append(w_psi[0])
                for child in node.children:
                    count_lots_on_S_psi4demand(child, S_list)
                return S_list

            S_list = []
            year_lots_list4S = []
            S_list = count_lots_on_S_psi4demand(root_node_out_opt, S_list)
            plan_year_st = year_st
            
            for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):
                year_lots4S = count_lots_yyyy(S_list, str(yyyy))
                year_lots_list4S.append(year_lots4S)
            
            #@241205 STOP NOT change "global_market_potential" at 2nd loading
            ## 値をインスタンス変数に保存
            #self.global_market_potential = year_lots_list4S[1]  


            print("NOT change #market_potential# at 2nd loading")
            print("self.market_potential", self.market_potential)




        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                filepath = os.path.join(directory, filename)
                print(f"Loading file: {filename}")
                if "outbound" in filename.lower():
                    self.outbound_data.append(pd.read_csv(filepath))
                elif "inbound" in filename.lower():
                    self.inbound_data.append(pd.read_csv(filepath))
        print("Outbound files loaded.")
        print("Inbound files loaded.")



        #@ STOP optimize processでは初期loadのcost_stracture設定で完了している
        #base_leaf = self.nodes_outbound[self.base_leaf_name]
        #
        #root_price = set_price_leaf2root(base_leaf,self.root_node_out_opt,100)
        #print("root_price", root_price)
        #set_value_chain_outbound(root_price, self.root_node_out_opt)
        

        self.view_nx_matlib()
        self.root.after(1000, self.show_psi_graph)

        #@241222@ STOP RUN
        # パラメータの初期化と更新を呼び出し
        self.initialize_parameters()


        def count_lots_on_S_psi4demand(node, S_list):

            # leaf_node末端市場の判定
            if node.children == []:  # 子nodeがないleaf nodeの場合

                # psi_listからS_listを生成する
                for w_psi in node.psi4demand:  # weeklyのSをS_listに集計

                    S_list.append(w_psi[0])

            else:
                pass

            for child in node.children:
                count_lots_on_S_psi4demand(child, S_list)

            return S_list


        S_list = []
        year_lots_list4S = []

        # treeを生成した直後なので、root_node_out_optが使える
        S_list = count_lots_on_S_psi4demand(root_node_out_opt, S_list)

            # 開始年を取得する
        plan_year_st = year_st  # 開始年のセット in main()要修正
        
        for yyyy in range(plan_year_st, plan_year_st + plan_range + 1):
        
            year_lots4S = count_lots_yyyy(S_list, str(yyyy))
        
            year_lots_list4S.append(year_lots4S)
        
            #        # 結果を出力
            #       #print(yyyy, " year carrying lots:", year_lots)
            #
            #    # 結果を出力
            #   #print(" year_lots_list:", year_lots_list)
        
            # an image of sample data
            #
            # 2023  year carrying lots: 0
            # 2024  year carrying lots: 2919
            # 2025  year carrying lots: 2914
            # 2026  year carrying lots: 2986
            # 2027  year carrying lots: 2942
            # 2028  year carrying lots: 2913
            # 2029  year carrying lots: 0
            #
            # year_lots_list4S: [0, 2919, 2914, 2986, 2942, 2913, 0]

            #@241124 CHECK

        #@241205 STOP NOT change "market_potential" at 2nd loading
        ## 値をインスタンス変数に保存
        #self.market_potential = year_lots_list4S[1]  

        #print("year_lots_list4S", year_lots_list4S)

        #self.global_market_potential = year_lots_list4S[1]

        #print("self.global_market_potential", self.global_market_potential)



        for filename in os.listdir(directory):

            if filename.endswith(".csv"):

                filepath = os.path.join(directory, filename)

                print(f"Loading file: {filename}")


                if "outbound" in filename.lower():
                    self.outbound_data.append(pd.read_csv(filepath))
                elif "inbound" in filename.lower():
                    self.inbound_data.append(pd.read_csv(filepath))
        print("Outbound files loaded.")
        print("Inbound files loaded.")
















# *************************
# PSI graph 
# *************************
    def show_psi(self, bound, layer):
        print("making PSI graph data...")
    
        week_start = 1
        week_end = self.plan_range * 53
    
        psi_data = []
    
        if bound not in ["outbound", "inbound"]:
            print("error: outbound or inbound must be defined for PSI layer")
            return
    
        if layer not in ["demand", "supply"]:
            print("error: demand or supply must be defined for PSI layer")
            return
    
        def traverse_nodes(node):
            for child in node.children:
                traverse_nodes(child)
            collect_psi_data(node, layer, week_start, week_end, psi_data)
    
        if bound == "outbound":
            traverse_nodes(self.root_node_outbound)
        else:
            traverse_nodes(self.root_node_inbound)
    
        fig, axs = plt.subplots(len(psi_data), 1, figsize=(5, len(psi_data) * 1))  # figsizeの高さをさらに短く設定
    
        if len(psi_data) == 1:
            axs = [axs]

        for ax, (node_name, revenue, profit, profit_ratio, line_plot_data_2I, bar_plot_data_3P, bar_plot_data_0S) in zip(axs, psi_data):
            ax2 = ax.twinx()
    
            ax.bar(line_plot_data_2I.index, line_plot_data_2I.values, color='r', alpha=0.6)
            ax.bar(bar_plot_data_3P.index, bar_plot_data_3P.values, color='g', alpha=0.6)
            ax2.plot(bar_plot_data_0S.index, bar_plot_data_0S.values, color='b')
    
            ax.set_ylabel('I&P Lots', fontsize=8)
            ax2.set_ylabel('S Lots', fontsize=8)
            ax.set_title(f'Node: {node_name} | REVENUE: {revenue:,} | PROFIT: {profit:,} | PROFIT_RATIO: {profit_ratio}%', fontsize=8)
    
        fig.tight_layout(pad=0.5)
    
        print("making PSI figure and widget...")

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
    
        canvas_psi = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas_psi.draw()
        canvas_psi.get_tk_widget().pack(fill=tk.BOTH, expand=True)





    def show_psi_graph(self):
        print("making PSI graph data...")

        week_start = 1
        week_end = self.plan_range * 53

        psi_data = []

        def traverse_nodes(node):
            for child in node.children:
                traverse_nodes(child)
            collect_psi_data(node, "demand", week_start, week_end, psi_data)

        # ***************************
        # ROOT HANDLE
        # ***************************
        traverse_nodes(self.root_node_outbound)

        fig, axs = plt.subplots(len(psi_data), 1, figsize=(5, len(psi_data) * 1))  # figsizeの高さをさらに短く設定

        if len(psi_data) == 1:
            axs = [axs]

        for ax, (node_name, revenue, profit, profit_ratio, line_plot_data_2I, bar_plot_data_3P, bar_plot_data_0S) in zip(axs, psi_data):
            ax2 = ax.twinx()

            ax.bar(line_plot_data_2I.index, line_plot_data_2I.values, color='r', alpha=0.6)
            ax.bar(bar_plot_data_3P.index, bar_plot_data_3P.values, color='g', alpha=0.6)
            ax2.plot(bar_plot_data_0S.index, bar_plot_data_0S.values, color='b')

            ax.set_ylabel('I&P Lots', fontsize=8)
            ax2.set_ylabel('S Lots', fontsize=8)
            ax.set_title(f'Node: {node_name} | REVENUE: {revenue:,} | PROFIT: {profit:,} | PROFIT_RATIO: {profit_ratio}%', fontsize=8)

            # Y軸の整数設定
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.tight_layout(pad=0.5)

        print("making PSI figure and widget...")

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        canvas_psi = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas_psi.draw()
        canvas_psi.get_tk_widget().pack(fill=tk.BOTH, expand=True)




    #@241225 marked revenueとprofitは、node classにインスタンスあり
    def show_psi_graph4opt(self):
        print("making PSI graph data...")

        week_start = 1
        week_end = self.plan_range * 53

        psi_data = []

        nodes_outbound = self.nodes_outbound  # node辞書{}

        def traverse_nodes(node_opt):
            for child in node_opt.children:
                print("show_psi_graph4opt child.name", child.name)
                traverse_nodes(child)
            node_out = nodes_outbound[node_opt.name]
            collect_psi_data_opt(node_opt, node_out, "supply", week_start, week_end, psi_data)

        # ***************************
        # change ROOT HANDLE
        # ***************************
        traverse_nodes(self.root_node_out_opt)

        fig, axs = plt.subplots(len(psi_data), 1, figsize=(5, len(psi_data) * 1))  # figsizeの高さをさらに短く設定

        if len(psi_data) == 1:
            axs = [axs]

        for ax, (node_name, revenue, profit, profit_ratio, line_plot_data_2I, bar_plot_data_3P, bar_plot_data_0S) in zip(axs, psi_data):
            ax2 = ax.twinx()

            ax.bar(line_plot_data_2I.index, line_plot_data_2I.values, color='r', alpha=0.6)
            ax.bar(bar_plot_data_3P.index, bar_plot_data_3P.values, color='g', alpha=0.6)
            ax2.plot(bar_plot_data_0S.index, bar_plot_data_0S.values, color='b')

            ax.set_ylabel('I&P Lots', fontsize=8)
            ax2.set_ylabel('S Lots', fontsize=8)
            ax.set_title(f'Node: {node_name} | REVENUE: {revenue:,} | PROFIT: {profit:,} | PROFIT_RATIO: {profit_ratio}%', fontsize=8)

            # Y軸の整数設定
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.tight_layout(pad=0.5)

        print("making PSI figure and widget...")

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        canvas_psi = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas_psi.draw()
        canvas_psi.get_tk_widget().pack(fill=tk.BOTH, expand=True)





