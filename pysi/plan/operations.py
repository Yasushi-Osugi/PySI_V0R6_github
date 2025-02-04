

# ******************************
# PSI_plan/planning_operation.py
# ******************************

# ****************************
# PSI planning operation on tree
# ****************************

def set_S2psi(node, pSi):

    # S_lots_listが辞書で、node.psiにセットする

    # print("len(node.psi4demand) = ", len(node.psi4demand) )
    # print("len(pSi) = ", len(pSi) )

    for w in range(len(pSi)):  # Sのリスト

        node.psi4demand[w][0].extend(pSi[w])



def calcS2P(node): # backward planning

    # **************************
    # Safety Stock as LT shift
    # **************************
    # leadtimeとsafety_stock_weekは、ここでは同じ

    # 同一node内なので、ssのみで良い
    shift_week = int(round(node.SS_days / 7))

    ## stop 同一node内でのLT shiftは無し
    ## SS is rounded_int_num
    # shift_week = node.leadtime +  int(round(node.SS_days / 7))

    # **************************
    # long vacation weeks
    # **************************
    lv_week = node.long_vacation_weeks

    # 同じnode内でのS to P の計算処理 # backward planning
    node.psi4demand = shiftS2P_LV(node.psi4demand, shift_week, lv_week)

    pass




def get_set_childrenP2S2psi(node, plan_range):

    for child in node.children:

        for w in range(node.leadtime, 53 * plan_range):

            # ******************
            # logistics LT switch
            # ******************
            # 物流をnodeとして定義する場合の表現 STOP
            # 子node childのP [3]のweek positionを親node nodeのS [0]にset
            # node.psi4demand[w][0].extend(child.psi4demand[w][3])

            # 物流をLT_shiftで定義する場合の表現 GO
            # childのPのweek positionをLT_shiftして、親nodeのS [0]にset
            ws = w - node.leadtime
            node.psi4demand[ws][0].extend(child.psi4demand[w][3])



# ******************************
# PSI_plan.demand_processing.py
# ******************************


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





# sliced df をcopyに変更
def make_lot_id_list_list(df_weekly, node_name):
    # 指定されたnode_nameがdf_weeklyに存在するか確認
    if node_name not in df_weekly["node_name"].values:
        return "Error: The specified node_name does not exist in df_weekly."

    # node_nameに基づいてデータを抽出
    df_node = df_weekly[df_weekly["node_name"] == node_name].copy()

    # 'iso_year'列と'iso_week'列を結合して新しいキーを作成
    df_node.loc[:, "iso_year_week"] = df_node["iso_year"].astype(str) + df_node[
        "iso_week"
    ].astype(str)

    # iso_year_weekでソート
    df_node = df_node.sort_values("iso_year_week")

    # lot_id_listのリストを生成
    pSi = [lot_id_list for lot_id_list in df_node["lot_id_list"]]

    return pSi



# dfを渡す
def set_df_Slots2psi4demand(node, df_weekly):
    # def set_psi_lists_postorder(node, node_psi_dict):


    #@240930
    #print("df_weekly@240930",df_weekly)

# an image of "df_weekly"
#
#df_weekly@240930      product_name node_name  ...  S_lot                   lot_id_list
#0          prod-A       CAN  ...      0                            []
#1          prod-A       CAN  ...      0                            []
#2          prod-A       CAN  ...      0                            []
#3          prod-A       CAN  ...      0                            []
#4          prod-A       CAN  ...      0                            []
#...           ...       ...  ...    ...                           ...
#1850       prod-A     SHA_N  ...      2  [SHA_N2024490, SHA_N2024491]
#1851       prod-A     SHA_N  ...      2  [SHA_N2024500, SHA_N2024501]
#1852       prod-A     SHA_N  ...      2  [SHA_N2024510, SHA_N2024511]
#1853       prod-A     SHA_N  ...      2  [SHA_N2024520, SHA_N2024521]
#1854       prod-A     SHA_N  ...      1                 [SHA_N202510]
#









    for child in node.children:

        set_df_Slots2psi4demand(child, df_weekly)

    # leaf_node末端市場の判定
    if node.children == []:  # 子nodeがないleaf nodeの場合

        # df_weeklyからnode.nameで、pSi[w]=lot_id_listとなるlistを作る
        # node.nameが存在しない場合はerror

        # nodeのSリスト pSi[w]を作る
        pSi = make_lot_id_list_list(df_weekly, node.name)

        #@STOP
        #print("make_lot_id_list_list node.name", node.name)
        #print("make_lot_id_list_list df_weekly", df_weekly)
        #print("make_lot_id_list_list pSi", pSi)


        ## probare 
        ##@240929
        #if node.name == "MUC_I":
        #    print("node.name pSi set_df_Slots2psi4demand")
        #    print("node.name pSi set_df_Slots2psi4demand",node.name, pSi)


#@240929 
# 1. lot_idの定義yyyywwnnに合せる
# 2. list position[]と"yyyyww"は、"start_year+week_no"と整合させる
# 3. find_yyyyww(start_year,week_no)でyyyywwを、find_week_no(start_year,yyyyww)
#    を、初期処理で、固定tableで用意する
#node.name pSi set_df_Slots2psi4demand MUC_I [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], ['MUC_I202410'], ['MUC_I2024100'], ['MUC_I2024110'], ['MUC_I2024120'], ['MUC_I2024130'], ['MUC_I2024140'], ['MUC_I2024150'], ['MUC_I2024160'], ['MUC_I2024170'], ['MUC_I2024180'], ['MUC_I2024190'], ['MUC_I202420'], ['MUC_I2024200'], ['MUC_I2024210'], ['MUC_I2024220'], ['MUC_I2024230'], ['MUC_I2024240'], ['MUC_I2024250'], ['MUC_I2024260'], ['MUC_I2024270'], ['MUC_I2024280'], ['MUC_I2024290'], ['MUC_I202430'], ['MUC_I2024300'], ['MUC_I2024310'], ['MUC_I2024320'], ['MUC_I2024330'], ['MUC_I2024340'], ['MUC_I2024350'], ['MUC_I2024360'], ['MUC_I2024370'], ['MUC_I2024380'], ['MUC_I2024390'], ['MUC_I202440'], ['MUC_I2024400'], ['MUC_I2024410'], ['MUC_I2024420'], ['MUC_I2024430'], ['MUC_I2024440'], ['MUC_I2024450'], ['MUC_I2024460'], ['MUC_I2024470'], ['MUC_I2024480'], ['MUC_I2024490'], ['MUC_I202450'], ['MUC_I2024500'], ['MUC_I2024510'], ['MUC_I2024520'], ['MUC_I202460'], ['MUC_I202470'], ['MUC_I202480'], ['MUC_I202490'], ['MUC_I202510']]









        # print("node.name pSi", node.name, pSi)
        # print("len(pSi) = ", len(pSi))

        # Sのリストをself.psi4demand[w][0].extend(pSi[w])
        node.set_S2psi(pSi)


        #@241124 probe psi4demand[][]
        print("241124 probe psi4demand[][]", node.psi4demand)


        # memo for animation
        # ココで、Sの初期セット状態とbackward shiftをanimationすると分りやすい
        # Sをセットしたら一旦、外に出て、Sの初期状態を表示すると動きが分かる
        # Sのbackward LD shiftを別途、処理する。

        # shifting S2P
        # shiftS2P_LV()は"lead time"と"安全在庫"のtime shift
        node.calcS2P()  # backward plan with postordering


    else:

        # 物流をnodeとして定義する場合は、メソッド修正get_set_childrenP2S2psi

        # logistic_LT shiftしたPをセットしてからgathering
        # 親nodeを見てlogistic_LT_shiftでP2Sを.extend(lots)すればgathering不要

        # ココは、calc_bw_psi処理として外出しする

        # gathering S and Setting S
        node.get_set_childrenP2S2psi(node.plan_range)

        # shifting S2P
        # shiftS2P_LV()は"lead time"と"安全在庫"のtime shift
        node.calcS2P()  # backward plan with postordering



# 同一node内のS2Pの処理
def shiftS2P_LV(psiS, shift_week, lv_week):  # LV:long vacations

    # ss = safety_stock_week
    sw = shift_week

    plan_len = len(psiS) - 1  # -1 for week list position

    for w in range(plan_len, sw, -1):  # backward planningで需要を降順でシフト

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        eta_plan = w - sw  # sw:shift week (includung safty stock)

        eta_shift = check_lv_week_bw(lv_week, eta_plan)  # ETA:Estimate Time Arrival

        # リスト追加 extend
        # 安全在庫とカレンダ制約を考慮した着荷予定週Pに、w週Sからoffsetする

        psiS[eta_shift][3].extend(psiS[w][0])  # P made by shifting S with

    return psiS

# ************************************
# checking constraint to inactive week , that is "Long Vacation"
# ************************************
def check_lv_week_bw(const_lst, check_week):

    num = check_week

    if const_lst == []:

        pass

    else:

        while num in const_lst:

            num -= 1

    return num


def check_lv_week_fw(const_lst, check_week):

    num = check_week

    if const_lst == []:

        pass

    else:

        while num in const_lst:

            num += 1

    return num








