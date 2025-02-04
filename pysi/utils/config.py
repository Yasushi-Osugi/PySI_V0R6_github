

"""
Configuration module for Global Weekly PSI Planner.

This module contains constants and configuration parameters used across the project.
"""

class Config:
    # General Settings
    APP_NAME = "Global Weekly PSI Planner"
    VERSION = "1.0.0"

    # File Paths
    DATA_DIRECTORY = "data/"
    OUTPUT_DIRECTORY = "output/"
    PROFILE_TREE_INBOUND = f"{DATA_DIRECTORY}profile_tree_inbound.csv"
    PROFILE_TREE_OUTBOUND = f"{DATA_DIRECTORY}profile_tree_outbound.csv"
    NODE_COST_TABLE_INBOUND = f"{DATA_DIRECTORY}node_cost_table_inbound.csv"
    NODE_COST_TABLE_OUTBOUND = f"{DATA_DIRECTORY}node_cost_table_outbound.csv"
    MONTHLY_DEMAND_FILE = f"{DATA_DIRECTORY}S_month_data.csv"

    # Default Parameters
    DEFAULT_LOT_SIZE = 2000
    DEFAULT_PLAN_RANGE = 5  # Default planning range in years
    DEFAULT_START_YEAR = 2025

    DEFAULT_PRE_PROC_LT = 13

    DEFAULT_MARKET_POTENTIAL = 10000
    DEFAULT_TARGET_SHARE = 0.5

    # Evaluation Parameters
    DEFAULT_PROFIT_RATIO = 0.6
    DEFAULT_INTEREST_RATE = 0.05  # Annual interest rate
    DEFAULT_WH_COST_RATIO = 0.01  # Warehouse cost ratio as a percentage of revenue

    # Visualization Settings
    PSI_GRAPH_TITLE = "PSI Graph"
    PSI_GRAPH_X_LABEL = "Weeks"
    PSI_GRAPH_Y_LABEL = "Values"

    # Logging Settings
    LOG_FILE = f"{OUTPUT_DIRECTORY}application.log"
    LOG_LEVEL = "DEBUG"

if __name__ == "__main__":
    print("Configuration module for Global Weekly PSI Planner")

