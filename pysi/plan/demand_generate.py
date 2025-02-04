
#### **`PSI_plan/demand_generate.py`**

import math
import pandas as pd
import numpy as np




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




def convert_monthly_to_weekly(df: pd.DataFrame, lot_size: int):
    """
    Convert monthly demand data to weekly ISO format with lot IDs and return additional metadata.

    Parameters:
        df (pd.DataFrame): Monthly demand data.
        lot_size (int): Lot size for allocation.

    Returns:
        Tuple[pd.DataFrame, int, int]: Weekly demand data, planning range, and starting year.
    """

    # デバッグ出力
    print("DataFrame type:", type(df))
    print("DataFrame columns:", df.columns if isinstance(df, pd.DataFrame) else "Not a DataFrame")
    print("DataFrame head:", df.head() if isinstance(df, pd.DataFrame) else "Not a DataFrame")


    # ** Check and extract plan range and starting year **


    ##@250118 STOP
    #plan_year_st = df["year"].min()
    #plan_year_end = df["year"].max()
    #plan_range = plan_year_end - plan_year_st + 1

    #@250118 RUN
    # *********************************
    # check_plan_range
    # *********************************
    plan_range, plan_year_st = check_plan_range(df)  # df is dataframe



    # ** Reshape data for processing **
    df = df.melt(
        id_vars=["product_name", "node_name", "year"],
        var_name="month",
        value_name="value",
    )
    df["month"] = df["month"].str[1:].astype(int)

    # ** Convert monthly data to daily data **
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

    # ** Aggregate data by ISO week **
    df_daily["iso_year"] = df_daily["date"].dt.isocalendar().year
    df_daily["iso_week"] = df_daily["date"].dt.isocalendar().week.astype(str).str.zfill(2)

    df_weekly = (
        df_daily.groupby(["product_name", "node_name", "iso_year", "iso_week"])["value"]
        .sum()
        .reset_index()
    )

    # ** Add lot-based calculations **
    df_weekly["S_lot"] = df_weekly["value"].apply(lambda x: math.ceil(x / lot_size))
    df_weekly["lot_id_list"] = df_weekly.apply(generate_lot_ids, axis=1)

    return df_weekly, plan_range, plan_year_st



def generate_lot_ids(row):
    """
    Generate lot IDs based on the row's data.

    Parameters:
        row (pd.Series): A row from the DataFrame.

    Returns:
        list: List of generated lot IDs.
    """
    lot_count = row["S_lot"]

    # "_" を削除した形式で生成
    return [f"{row['node_name']}{row['iso_year']}{str(row['iso_week']).zfill(2)}{i+1:03d}" for i in range(lot_count)]

def generate_lot_ids_OLD(row):
    """
    Generate lot IDs based on the row's data.

    Parameters:
        row (pd.Series): A row from the DataFrame.

    Returns:
        list: List of generated lot IDs.
    """
    lot_count = row["S_lot"]

    # "_" を削除した形式で生成
    return [f"{row['node_name']}{row['iso_year']}{row['iso_week']}{i+1:03d}" for i in range(lot_count)]


# 3桁を4桁にする時は、gui下の"app.py"にある"def extract_node_name()"も修正する
#lot_IDが3桁の場合{i+1:03d}に、 9文字を削る
#lot_IDが4桁の場合{i+1:04d}に、10文字を削る
#
#def extract_node_name(stringA):
#    """
#    Extract the node name from a string by removing the last 9 characters (YYYYWWNNN).
#
#    Parameters:
#        stringA (str): Input string (e.g., "LEAF01202601001").
#
#    Returns:
#        str: Node name (e.g., "LEAF01").
#    """
#    if len(stringA) > 9:
#        # 最後の9文字を削除して返す #deep relation on "def generate_lot_ids()"
#        return stringA[:-9]
#    else:
#        # 文字列が9文字以下の場合、削除せずそのまま返す（安全策）
#        return stringA





def generate_lot_ids_OLD(row):
    """
    Generate lot IDs based on the row's data.

    Parameters:
        row (pd.Series): A row from the DataFrame.

    Returns:
        list: List of generated lot IDs.
    """
    lot_count = row["S_lot"]

    return [f"{row['node_name']}_{row['iso_year']}{row['iso_week']}_{i+1}" for i in range(lot_count)]

