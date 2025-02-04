#network_tree250114.py


# network/tree.py
from typing import List, Dict, Optional

from collections import defaultdict

from pysi.utils.file_io import read_tree_file
#from some_module import Node  # Node�N���X��K�؂ȏꏊ����C���|�[�g

#from pysi.plan.demand_processing import *
#from plan.demand_processing import shiftS2P_LV

from pysi.plan.operations import *
#from pysi.plan.operations import calcS2P, set_S2psi, get_set_childrenP2S2psi





class Node:
    def __init__(self, name: str):
        self.name = name
        self.children: List['Node'] = []
        self.parent: Optional['Node'] = None

        self.depth = 0
        self.width = 0
        self.lot_size = 1  # default setting
        self.psi = []  # Placeholder for PSI data

        self.iso_week_demand = None  # Original demand converted to ISO week
        self.psi4demand = None
        self.psi4supply = None
        self.psi4couple = None
        self.psi4accume = None

        self.plan_range = 1
        self.plan_year_st = 2025

        self.safety_stock_week = 0
        self.long_vacation_weeks = []

        # For NetworkX
        self.leadtime = 1  # same as safety_stock_week
        self.nx_demand = 1  # weekly average demand by lot
        self.nx_weight = 1  # move_cost_all_to_nodeB (from nodeA to nodeB)
        self.nx_capacity = 1  # lot by lot

        # Evaluation
        self.decoupling_total_I = []  # total Inventory all over the plan

        # Position
        self.longitude = None
        self.latitude = None

        # "lot_counts" is the bridge PSI2EVAL
        self.lot_counts = [0 for x in range(0, 53 * self.plan_range)]
        self.lot_counts_all = 0  # sum(self.lot_counts)

        # Settings for cost-profit evaluation parameter
        self.LT_boat = 1
        self.SS_days = 7
        self.HS_code = ""
        self.customs_tariff_rate = 0
        self.tariff_on_price = 0
        self.price_elasticity = 0




        # ******************************
        # evaluation data initialise rewards���v�Z�̏�����
        # ******************************

        # ******************************
        # Profit_Ratio #float
        # ******************************
        self.eval_profit_ratio = Profit_Ratio = 0.6

        # Revenue, Profit and Costs
        self.eval_revenue = 0
        self.eval_profit = 0



        self.eval_PO_cost = 0
        self.eval_P_cost = 0
        self.eval_WH_cost = 0
        self.eval_SGMC = 0
        self.eval_Dist_Cost = 0



        # ******************************
        # set_EVAL_cash_in_data #list for 53weeks * 5 years # 5�N��z��
        # *******************************
        self.Profit = Profit = [0 for i in range(53 * self.plan_range)]
        self.Week_Intrest = Week_Intrest = [0 for i in range(53 * self.plan_range)]
        self.Cash_In = Cash_In = [0 for i in range(53 * self.plan_range)]
        self.Shipped_LOT = Shipped_LOT = [0 for i in range(53 * self.plan_range)]
        self.Shipped = Shipped = [0 for i in range(53 * self.plan_range)]

        # ******************************
        # set_EVAL_cash_out_data #list for 54 weeks
        # ******************************

        self.SGMC = SGMC = [0 for i in range(53 * self.plan_range)]
        self.PO_manage = PO_manage = [0 for i in range(53 * self.plan_range)]
        self.PO_cost = PO_cost = [0 for i in range(53 * self.plan_range)]
        self.P_unit = P_unit = [0 for i in range(53 * self.plan_range)]
        self.P_cost = P_cost = [0 for i in range(53 * self.plan_range)]

        self.I = I = [0 for i in range(53 * self.plan_range)]

        self.I_unit = I_unit = [0 for i in range(53 * self.plan_range)]
        self.WH_cost = WH_cost = [0 for i in range(53 * self.plan_range)]
        self.Dist_Cost = Dist_Cost = [0 for i in range(53 * self.plan_range)]




        # Cost structure demand
        self.price_sales_shipped = 0
        self.cost_total = 0
        self.profit = 0
        self.marketing_promotion = 0
        self.sales_admin_cost = 0
        self.SGA_total = 0
        self.custom_tax = 0
        self.tax_portion = 0
        self.logistics_costs = 0
        self.warehouse_cost = 0
        self.direct_materials_costs = 0
        self.purchase_total_cost = 0
        self.prod_indirect_labor = 0
        self.prod_indirect_others = 0
        self.direct_labor_costs = 0
        self.depreciation_others = 0
        self.manufacturing_overhead = 0

        # Profit accumulated root to node
        self.cs_profit_accume = 0

        # Cost Structure
        self.cs_price_sales_shipped = 0
        self.cs_cost_total = 0
        self.cs_profit = 0
        self.cs_marketing_promotion = 0
        self.cs_sales_admin_cost = 0
        self.cs_SGA_total = 0
        self.cs_custom_tax = 0
        self.cs_tax_portion = 0
        self.cs_logistics_costs = 0
        self.cs_warehouse_cost = 0
        self.cs_direct_materials_costs = 0
        self.cs_purchase_total_cost = 0
        self.cs_prod_indirect_labor = 0
        self.cs_prod_indirect_others = 0
        self.cs_direct_labor_costs = 0
        self.cs_depreciation_others = 0
        self.cs_manufacturing_overhead = 0

        # Evaluated cost = Cost Structure X lot_counts
        self.eval_cs_price_sales_shipped = 0  # revenue
        self.eval_cs_cost_total = 0  # cost
        self.eval_cs_profit = 0  # profit
        self.eval_cs_marketing_promotion = 0
        self.eval_cs_sales_admin_cost = 0
        self.eval_cs_SGA_total = 0
        self.eval_cs_custom_tax = 0
        self.eval_cs_tax_portion = 0
        self.eval_cs_logistics_costs = 0
        self.eval_cs_warehouse_cost = 0
        self.eval_cs_direct_materials_costs = 0
        self.eval_cs_purchase_total_cost = 0
        self.eval_cs_prod_indirect_labor = 0
        self.eval_cs_prod_indirect_others = 0
        self.eval_cs_direct_labor_costs = 0
        self.eval_cs_depreciation_others = 0
        self.eval_cs_manufacturing_overhead = 0

        # Shipped lots count W / M / Q / Y / LifeCycle
        self.shipped_lots_W = []  # 53*plan_range
        self.shipped_lots_M = []  # 12*plan_range
        self.shipped_lots_Q = []  # 4*plan_range
        self.shipped_lots_Y = []  # 1*plan_range
        self.shipped_lots_L = []  # 1  # lifecycle a year

        # Planned Amount
        self.amt_price_sales_shipped = []
        self.amt_cost_total = []
        self.amt_profit = []
        self.amt_marketing_promotion = []
        self.amt_sales_admin_cost = []
        self.amt_SGA_total = []
        self.amt_custom_tax = []
        self.amt_tax_portion = []
        self.amt_logistiamt_costs = []
        self.amt_warehouse_cost = []
        self.amt_direct_materials_costs = []
        self.amt_purchase_total_cost = []
        self.amt_prod_indirect_labor = []
        self.amt_prod_indirect_others = []
        self.amt_direct_labor_costs = []
        self.amt_depreciation_others = []
        self.amt_manufacturing_overhead = []

        # Shipped amt W / M / Q / Y / LifeCycle
        self.shipped_amt_W = []  # 53*plan_range
        self.shipped_amt_M = []  # 12*plan_range
        self.shipped_amt_Q = []  # 4*plan_range
        self.shipped_amt_Y = []  # 1*plan_range
        self.shipped_amt_L = []  # 1  # lifecycle a year

        # Control FLAGs
        self.cost_standard_flag = 0
        self.PSI_graph_flag = "OFF"
        self.buffering_stock_flag = "OFF"

        self.revenue = 0
        self.profit = 0






    def add_child(self, child: 'Node'):
        """Add a child node to the current node."""
        self.children.append(child)
        child.parent = self

    def set_depth(self, depth: int):
        """Recursively set the depth of the node and its children."""
        self.depth = depth
        for child in self.children:
            child.set_depth(depth + 1)

    def print_tree(self, level: int = 0):
        """Print the tree structure starting from the current node."""
        print("  " * level + f"Node: {self.name}")
        for child in self.children:
            child.print_tree(level + 1)



    # ********************************
    # �R�R�ő������Z�b�g@240417
    # ********************************
    def set_attributes(self, row):

        #print("set_attributes(self, row):", row)
        # self.lot_size = int(row[3])
        # self.leadtime = int(row[4])  # �O��:SS=0
        # self.long_vacation_weeks = eval(row[5])

        self.lot_size = int(row["lot_size"])

        # ********************************
        # with using NetworkX
        # ********************************

        # weight��capacity�́Aedge=(node_A,node_B)�̑�����node�ň�ӂł͂Ȃ�

        self.leadtime = int(row["leadtime"])  # �O��:SS=0 # "weight"4NetworkX
        self.capacity = int(row["process_capa"])  # "capacity"4NetworkX

        self.long_vacation_weeks = eval(row["long_vacation_weeks"])

        # **************************
        # BU_SC_node_profile     business_unit_supplychain_node
        # **************************

        # @240421 �@�B�w�K�̃t���O��stop
        ## **************************
        ## plan_basic_parameter ***sequencing is TEMPORARY
        ## **************************
        #        self.PlanningYear           = row['plan_year']
        #        self.plan_engine            = row['plan_engine']
        #        self.reward_sw              = row['reward_sw']

        # ���i�KPSI�̃t���O��stop
        ## ***************************
        ## business unit identify
        ## ***************************
        #        self.product_name           = row['product_name']
        #        self.SC_tree_id             = row['SC_tree_id']
        #        self.node_from              = row['node_from']
        #        self.node_to                = row['node_to']


        # ***************************
        # �R�R����cost-profit evaluation �p�̑����Z�b�g
        # ***************************
        self.LT_boat = float(row["LT_boat"])



        self.SS_days = float(row["SS_days"])


        print("row[ customs_tariff_rate ]", row["customs_tariff_rate"])



        self.HS_code              = str(row["HS_code"])
        self.customs_tariff_rate  = float(row["customs_tariff_rate"])
        self.price_elasticity     = float(row["price_elasticity"])



        self.cost_standard_flag   = float(row["cost_standard_flag"])
        self.PSI_graph_flag       = str(row["PSI_graph_flag"])
        self.buffering_stock_flag = str(row["buffering_stock_flag"])

        self.base_leaf = None






    def set_parent(self):
        # def set_parent(self, node):

        # tree��H��Ȃ���e�m�[�h��T��
        if self.children == []:
            pass
        else:
            for child in self.children:
                child.parent = self
                # child.parent = node




    def set_cost_attr(
        self,
        price_sales_shipped,
        cost_total,
        profit,
        marketing_promotion=None,
        sales_admin_cost=None,
        SGA_total=None,
        custom_tax=None,
        tax_portion=None,
        logistics_costs=None,
        warehouse_cost=None,
        direct_materials_costs=None,
        purchase_total_cost=None,
        prod_indirect_labor=None,
        prod_indirect_others=None,
        direct_labor_costs=None,
        depreciation_others=None,
        manufacturing_overhead=None,
    ):

        # self.node_name = node_name # node_name is STOP
        self.price_sales_shipped = price_sales_shipped
        self.cost_total = cost_total
        self.profit = profit
        self.marketing_promotion = marketing_promotion
        self.sales_admin_cost = sales_admin_cost
        self.SGA_total = SGA_total
        self.custom_tax = custom_tax
        self.tax_portion = tax_portion
        self.logistics_costs = logistics_costs
        self.warehouse_cost = warehouse_cost
        self.direct_materials_costs = direct_materials_costs
        self.purchase_total_cost = purchase_total_cost
        self.prod_indirect_labor = prod_indirect_labor
        self.prod_indirect_others = prod_indirect_others
        self.direct_labor_costs = direct_labor_costs
        self.depreciation_others = depreciation_others
        self.manufacturing_overhead = manufacturing_overhead

    def print_cost_attr(self):

        # self.node_name = node_name # node_name is STOP
        print("self.price_sales_shipped", self.price_sales_shipped)
        print("self.cost_total", self.cost_total)
        print("self.profit", self.profit)
        print("self.marketing_promotion", self.marketing_promotion)
        print("self.sales_admin_cost", self.sales_admin_cost)
        print("self.SGA_total", self.SGA_total)
        print("self.custom_tax", self.custom_tax)
        print("self.tax_portion", self.tax_portion)
        print("self.logistics_costs", self.logistics_costs)
        print("self.warehouse_cost", self.warehouse_cost)
        print("self.direct_materials_costs", self.direct_materials_costs)
        print("self.purchase_total_cost", self.purchase_total_cost)
        print("self.prod_indirect_labor", self.prod_indirect_labor)
        print("self.prod_indirect_others", self.prod_indirect_others)
        print("self.direct_labor_costs", self.direct_labor_costs)
        print("self.depreciation_others", self.depreciation_others)
        print("self.manufacturing_overhead", self.manufacturing_overhead)





    def set_plan_range_lot_counts(self, plan_range, plan_year_st):

        # print("node.plan_range", self.name, self.plan_range)

        self.plan_range = plan_range
        self.plan_year_st = plan_year_st

        self.lot_counts = [0 for x in range(0, 53 * self.plan_range)]


        for child in self.children:

            child.set_plan_range_lot_counts(plan_range, plan_year_st)







# ****************************
# PSI planning operation on tree
# ****************************

    def set_S2psi(self, pSi):

        # S_lots_list�������ŁAnode.psi�ɃZ�b�g����

        # print("len(self.psi4demand) = ", len(self.psi4demand) )
        # print("len(pSi) = ", len(pSi) )

        for w in range(len(pSi)):  # S�̃��X�g

            self.psi4demand[w][0].extend(pSi[w])



    def calcS2P(self): # backward planning

        # **************************
        # Safety Stock as LT shift
        # **************************
        # leadtime��safety_stock_week�́A�����ł͓���

        # ����node���Ȃ̂ŁAss�݂̂ŗǂ�
        shift_week = int(round(self.SS_days / 7))

        ## stop ����node���ł�LT shift�͖���
        ## SS is rounded_int_num
        # shift_week = self.leadtime +  int(round(self.SS_days / 7))

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # ����node���ł�S to P �̌v�Z���� # backward planning
        self.psi4demand = shiftS2P_LV(self.psi4demand, shift_week, lv_week)

        pass





    def get_set_childrenP2S2psi(self, plan_range):

        for child in self.children:

            for w in range(self.leadtime, 53 * plan_range):

                # ******************
                # logistics LT switch
                # ******************
                # ������node�Ƃ��Ē�`����ꍇ�̕\�� STOP
                # �qnode child��P [3]��week position��enode self��S [0]��set
                # self.psi4demand[w][0].extend(child.psi4demand[w][3])

                # ������LT_shift�Œ�`����ꍇ�̕\�� GO
                # child��P��week position��LT_shift���āA�enode��S [0]��set
                ws = w - self.leadtime
                self.psi4demand[ws][0].extend(child.psi4demand[w][3])




    # ******************
    # for debug
    # ******************
    def show_sum_cs(self):

        cs_sum = 0

        cs_sum = (
            self.cs_direct_materials_costs
            + self.cs_marketing_promotion
            + self.cs_sales_admin_cost
            + self.cs_tax_portion
            + self.cs_logistics_costs
            + self.cs_warehouse_cost
            + self.cs_prod_indirect_labor
            + self.cs_prod_indirect_others
            + self.cs_direct_labor_costs
            + self.cs_depreciation_others
            + self.cs_profit
        )

       #print("cs_sum", self.name, cs_sum)



    # ******************************
    # evaluation 
    # ******************************


    def set_lot_counts(self):

        plan_len = 53 * self.plan_range

        for w in range(0, plan_len):  ### �ȉ���i+1��1�T�X�^�[�g = W1,W2,W3,,
            self.lot_counts[w] = len(self.psi4supply[w][3])  # psi[w][3]=PO

        self.lot_counts_all = sum(self.lot_counts)





    def EvalPlanSIP_cost(self):

        L = self.lot_counts_all    # node�̑S���b�g�� # psi[w][3]=PO
    
        # evaluated cost = Cost Structure X lot_counts
        self.eval_cs_price_sales_shipped    = L * self.cs_price_sales_shipped
        self.eval_cs_cost_total             = L * self.cs_cost_total
        self.eval_cs_profit                 = L * self.cs_profit
        self.eval_cs_marketing_promotion    = L * self.cs_marketing_promotion
        self.eval_cs_sales_admin_cost       = L * self.cs_sales_admin_cost
        self.eval_cs_SGA_total              = L * self.cs_SGA_total
        self.eval_cs_custom_tax             = L * self.cs_custom_tax
        self.eval_cs_tax_portion            = L * self.cs_tax_portion
        self.eval_cs_logistics_costs        = L * self.cs_logistics_costs
        self.eval_cs_warehouse_cost         = L * self.cs_warehouse_cost
        self.eval_cs_direct_materials_costs = L * self.cs_direct_materials_costs    
        self.eval_cs_purchase_total_cost    = L * self.cs_purchase_total_cost
        self.eval_cs_prod_indirect_labor    = L * self.cs_prod_indirect_labor
        self.eval_cs_prod_indirect_others   = L * self.cs_prod_indirect_others
        self.eval_cs_direct_labor_costs     = L * self.cs_direct_labor_costs
        self.eval_cs_depreciation_others    = L * self.cs_depreciation_others
        self.eval_cs_manufacturing_overhead = L * self.cs_manufacturing_overhead    
    
        # �݌ɌW���̌v�Z
        I_total_qty_planned, I_total_qty_init = self.I_lot_counts_all() 
    
        if I_total_qty_init == 0:

            I_cost_coeff = 0

        else:

            I_cost_coeff =  I_total_qty_planned / I_total_qty_init
    
        print("self.name",self.name)
        print("I_total_qty_planned", I_total_qty_planned)
        print("I_total_qty_init", I_total_qty_init)
        print("I_cost_coeff", I_cost_coeff)
    
        # �݌ɂ̑����W�����|���ăZ�b�g

        print("self.eval_cs_warehouse_cost", self.eval_cs_warehouse_cost)

        self.eval_cs_warehouse_cost *= ( 1 + I_cost_coeff )

        print("self.eval_cs_warehouse_cost", self.eval_cs_warehouse_cost)

    
        self.eval_cs_cost_total = (

            self.eval_cs_marketing_promotion + 
            self.eval_cs_sales_admin_cost + 
            #self.eval_cs_SGA_total + 

            #self.eval_cs_custom_tax + 
            self.eval_cs_tax_portion + 
            self.eval_cs_logistics_costs + 
            self.eval_cs_warehouse_cost + 
            self.eval_cs_direct_materials_costs + 
            #self.eval_cs_purchase_total_cost + 

            self.eval_cs_prod_indirect_labor + 
            self.eval_cs_prod_indirect_others + 
            self.eval_cs_direct_labor_costs + 
            self.eval_cs_depreciation_others #@END + 
            #self.eval_cs_manufacturing_overhead
        )
    
        # profit = revenue - cost
        self.eval_cs_profit = self.eval_cs_price_sales_shipped - self.eval_cs_cost_total
    
        return self.eval_cs_price_sales_shipped, self.eval_cs_profit







    # *****************************
    # ������CPU_LOTs�𒊏o����
    # *****************************
    def extract_CPU(self, csv_writer):

        plan_len = 53 * self.plan_range  # �v�撷���Z�b�g

        # w=1���璊�o����

        # starting_I = 0 = w-1 / ending_I=plan_len
        for w in range(1, plan_len):

            # for w in range(1,54):   #starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4supply[w][0]

            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # ***************************
            # write CPU
            # ***************************
            #
            # ISO_week_no,
            # CPU_lot_id,
            # S-I-P�敪,
            # node���W(longitude, latitude),
            # step(����=���i��),
            # lot_size
            # ***************************

            # ***************************
            # write "s" CPU
            # ***************************
            for step_no, lot_id in enumerate(s):

                # lot_id���v��TYYYYWW�Ń��j�[�N�ɂ���
                lot_id_yyyyww = lot_id + str(self.plan_year_st) + str(w).zfill(3)

                CPU_row = [
                    w,
                    lot_id_yyyyww,
                    "s",
                    self.name,
                    self.longitude,
                    self.latitude,
                    step_no,
                    self.lot_size,
                ]

                csv_writer.writerow(CPU_row)

            # ***************************
            # write "i1" CPU
            # ***************************
            for step_no, lot_id in enumerate(i1):

                # lot_id���v��TYYYYWW�Ń��j�[�N�ɂ���
                lot_id_yyyyww = lot_id + str(self.plan_year_st) + str(w).zfill(3)

                CPU_row = [
                    w,
                    lot_id_yyyyww,
                    "i1",
                    self.name,
                    self.longitude,
                    self.latitude,
                    step_no,
                    self.lot_size,
                ]

                csv_writer.writerow(CPU_row)

            # ***************************
            # write "p" CPU
            # ***************************
            for step_no, lot_id in enumerate(p):

                # lot_id���v��TYYYYWW�Ń��j�[�N�ɂ���
                lot_id_yyyyww = lot_id + str(self.plan_year_st) + str(w).zfill(3)

                CPU_row = [
                    w,
                    lot_id_yyyyww,
                    "p",
                    self.name,
                    self.longitude,
                    self.latitude,
                    step_no,
                    self.lot_size,
                ]

                csv_writer.writerow(CPU_row)




    # ******************************
    # planning operation on tree
    # ******************************

    # ******************************
    # in or out    : root_node_outbound
    # plan layer   : demand layer
    # node order   : preorder # Leaf2Root
    # time         : Foreward
    # calculation  : PS2I
    # ******************************

    def calcPS2I4demand(self):

        # psiS2P = self.psi4demand # copy�����ɁA���ڂ����

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4demand)

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len
            # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4demand[w][0]
            co = self.psi4demand[w][1]

            i0 = self.psi4demand[w - 1][2]
            i1 = self.psi4demand[w][2]

            p = self.psi4demand[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # �O�T�݌ɂƓ��T���ו� availables

            # �����ŁA�����̍݌ɁAS�o��=����𑀍삵�Ă���
            # S�o��=����𖾎��I��log�ɂ��āA����Ƃ��ċL�^���A�\�����鏈��
            # �o�ׂ��ꂽS=����A�݌�I�A���o��CO�̏W���𐳂����\������

            # ���m�������ɑ���u�� #@240909�R���ł͂Ȃ���S����

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4demand[w][2] = i1 = diff_list



    def calcPS2I4supply(self):

        # psiS2P = self.psi4demand # copy�����ɁA���ڂ����

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4supply)

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len
            # for w in range(1,54): # starting_I = 0 = w-1 / ending_I = 53

            s = self.psi4supply[w][0]
            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # �O�T�݌ɂƓ��T���ו� availables

            # memo �����ŁA�����̍݌ɁAS�o��=����𑀍삵�Ă���
            # S�o��=����𖾎��I��log�ɂ��āA����Ƃ��ċL�^���A�\�����鏈��
            # �o�ׂ��ꂽS=����A�݌�I�A���o��CO�̏W���𐳂����\������

            # ���m�������ɑ���u��

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4supply[w][2] = i1 = diff_list

            # ************************************
            # probare a lot checking process
            # ************************************
            
            if self.name == "MUC_N":

                if w in [53,54,55,56,57]:

                    print("s, co, i0, i1, p ", w )
                    print("s" , w, s )
                    print("co", w, co)
                    print("i0", w, i0)
                    print("i1", w, i1)
                    print("p" , w, p )


    def calcPS2I_decouple4supply(self):

        # psiS2P = self.psi4demand # copy�����ɁA���ڂ����

        plan_len = 53 * self.plan_range
        # plan_len = len(self.psi4supply)

        # demand plan��S���o�׎w�����=PULL SIGNAL�Ƃ��āAsupply planS�ɃZ�b�g

        for w in range(0, plan_len):
            # for w in range(1,plan_len):

            # pointer�Q�Ƃ��Ă��Ȃ���? �����I�Ƀf�[�^��n���ɂ�?

            self.psi4supply[w][0] = self.psi4demand[w][
                0
            ].copy()  # copy data using copy() method

            # self.psi4supply[w][0]    = self.psi4demand[w][0] # PULL replaced

            # checking pull data
            # show_psi_graph(root_node_outbound,"supply", "HAM", 0, 300 )
            # show_psi_graph(root_node_outbound,"supply", node_show, 0, 300 )

        for w in range(1, plan_len):  # starting_I = 0 = w-1 / ending_I =plan_len

            # demand planS��supplyS�ɃR�s�[�ς�
            s = self.psi4supply[w][0]  # PUSH supply S

            co = self.psi4supply[w][1]

            i0 = self.psi4supply[w - 1][2]
            i1 = self.psi4supply[w][2]

            p = self.psi4supply[w][3]

            # *********************
            # # I(n-1)+P(n)-S(n)
            # *********************

            work = i0 + p  # �O�T�݌ɂƓ��T���ו� availables

            # memo �����ŁA�����̍݌ɁAS�o��=����𑀍삵�Ă���
            # S�o��=����𖾎��I��log�ɂ��āA����Ƃ��ċL�^���A�\�����鏈��
            # �o�ׂ��ꂽS=����A�݌�I�A���o��CO�̏W���𐳂����\������

            # ���m�������ɑ���u��

            diff_list = [x for x in work if x not in s]  # I(n-1)+P(n)-S(n)

            self.psi4supply[w][2] = i1 = diff_list




    def calcS2P(self): # backward planning

        # **************************
        # Safety Stock as LT shift
        # **************************
        # leadtime��safety_stock_week�́A�����ł͓���

        # ����node���Ȃ̂ŁAss�݂̂ŗǂ�
        shift_week = int(round(self.SS_days / 7))

        ## stop ����node���ł�LT shift�͖���
        ## SS is rounded_int_num
        # shift_week = self.leadtime +  int(round(self.SS_days / 7))

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # ����node���ł�S to P �̌v�Z���� # backward planning
        self.psi4demand = shiftS2P_LV(self.psi4demand, shift_week, lv_week)

        pass




    def calcS2P_4supply(self):    # "self.psi4supply"
        # **************************
        # Safety Stock as LT shift
        # **************************
        # leadtime��safety_stock_week�́A�����ł͓���

        # ����node���Ȃ̂ŁAss�݂̂ŗǂ�
        shift_week = int(round(self.SS_days / 7))

        ## stop ����node���ł�LT shift�͖���
        ## SS is rounded_int_num
        # shift_week = self.leadtime +  int(round(self.SS_days / 7))

        # **************************
        # long vacation weeks
        # **************************
        lv_week = self.long_vacation_weeks

        # S to P �̌v�Z����
        self.psi4supply = shiftS2P_LV_replace(self.psi4supply, shift_week, lv_week)

        pass




    def set_plan_range_lot_counts(self, plan_range, plan_year_st):

        # print("node.plan_range", self.name, self.plan_range)

        self.plan_range = plan_range
        self.plan_year_st = plan_year_st

        self.lot_counts = [0 for x in range(0, 53 * self.plan_range)]


        for child in self.children:

            child.set_plan_range_lot_counts(plan_range, plan_year_st)





    def I_lot_counts_all(self):
        lot_all_supply = 0
        lot_all_demand = 0

        plan_len = 53 * self.plan_range

        lot_counts_I_supply = [0] * plan_len
        lot_counts_I_demand = [0] * plan_len

        #@241129 DEBUG DUMP TEST self.psi4supply
        #if self.name == "HAM":
        #    print("self.psi4supply",self.psi4supply)




        for w in range(plan_len):  ### �ȉ���i+1��1�T�X�^�[�g = W1,W2,W3,,
            lot_counts_I_supply[w] = len(self.psi4supply[w][2])  # psi[w][2]=I
            lot_counts_I_demand[w] = len(self.psi4demand[w][2])  # psi[w][2]=I


        #@241129 DUMP TEST self.psi4supply
        if self.name == "HAM":
            print("lot_counts_I_supply",lot_counts_I_supply)



        lot_all_supply = sum(lot_counts_I_supply)
        lot_all_demand = sum(lot_counts_I_demand)

        return lot_all_supply, lot_all_demand








# ****************************
# after demand leveling / planning outbound supply
# ****************************
def shiftS2P_LV_replace(psiS, shift_week, lv_week):  # LV:long vacations

    # ss = safety_stock_week
    sw = shift_week

    plan_len = len(psiS) - 1  # -1 for week list position

    for w in range(plan_len):  # foreward planning��supply��p [w][3]��������

        # psiS[w][0] = [] # S active

        psiS[w][1] = []  # CO
        psiS[w][2] = []  # I
        psiS[w][3] = []  # P

    for w in range(plan_len, sw, -1):  # backward planning��supply���~���ŃV�t�g

        # my_list = [1, 2, 3, 4, 5]
        # for i in range(2, len(my_list)):
        #    my_list[i] = my_list[i-1] + my_list[i-2]

        # 0:S
        # 1:CO
        # 2:I
        # 3:P

        eta_plan = w - sw  # sw:shift week ( including safty stock )

        eta_shift = check_lv_week_bw(lv_week, eta_plan)  # ETA:Eatimate Time Arrival

        # ���X�g�ǉ� extend
        # ���S�݌ɂƃJ�����_������l���������ח\��TP�ɁAw�TS����offset����
        psiS[eta_shift][3].extend(psiS[w][0])  # P made by shifting S with

    return psiS






# ****************************
# PSI planning demand
# ****************************
def calc_all_psi2i4demand(node):

    node.calcPS2I4demand()

    for child in node.children:

        calc_all_psi2i4demand(child)



# ****************************
# tree positioing
# ****************************
def set_positions_recursive(node, width_tracker):
    for child in node.children:
        child.depth = node.depth + 1
        child.width = width_tracker[child.depth]
        width_tracker[child.depth] += 1
        set_positions_recursive(child, width_tracker)

def adjust_positions(node):
    if not node.children:
        return node.width

    children_y_min = min(adjust_positions(child) for child in node.children)
    children_y_max = max(adjust_positions(child) for child in node.children)
    node.width = (children_y_min + children_y_max) / 2

    for i, child in enumerate(node.children):
        child.width += i * 0.1

    return node.width

def set_positions(root):
    width_tracker = [0] * 100
    set_positions_recursive(root, width_tracker)
    adjust_positions(root)




def set_node_costs(cost_table, nodes):
    """
    Set cost attributes for nodes based on the given cost table.

    Parameters:
        cost_table (pd.DataFrame): DataFrame containing cost data.
        nodes (dict): Dictionary of node instances.

    Returns:
        None
    """
    df_transposed = cost_table.transpose()

    rows = df_transposed.iterrows()
    next(rows)  # Skip the header row

    for index, row in rows:
        node_name = index
        try:
            node = nodes[node_name]
            node.set_cost_attr(*row)
            node.print_cost_attr()
        except KeyError:
            print(f"Warning: {node_name} not found in nodes. Continuing with next item.")




def set_parent_all(node):
    # preordering

    if node.children == []:
        pass
    else:
        node.set_parent()  # ���̒��Ŏqnode�����Đe��������B
        # def set_parent(self)

    for child in node.children:

        set_parent_all(child)




def print_parent_all(node):
    # preordering

    if node.children == []:
        pass
    else:
        print("node.parent and children", node.name, node.children)

    for child in node.children:

        print("child and parent", child.name, node.name)

        print_parent_all(child)





def build_tree_from_dict(tree_dict: Dict[str, List[str]]) -> Node:
    """
    Build a tree structure from a dictionary.

    Parameters:
        tree_dict (Dict[str, List[str]]): A dictionary where keys are parent node names
                                         and values are lists of child node names.

    Returns:
        Node: The root node of the constructed tree.
    """
    nodes: Dict[str, Node] = {}

    # Create all nodes
    for parent, children in tree_dict.items():
        if parent not in nodes:
            nodes[parent] = Node(parent)
        for child in children:
            if child not in nodes:
                nodes[child] = Node(child)

    # Link nodes
    for parent, children in tree_dict.items():
        for child in children:
            nodes[parent].add_child(nodes[child])

    # Assume the root is the one without a parent
    root_candidates = set(nodes.keys()) - {child for children in tree_dict.values() for child in children}
    if len(root_candidates) != 1:
        raise ValueError("Tree must have exactly one root")

    root_name = root_candidates.pop()
    root = nodes[root_name]
    root.set_depth(0)
    return root






def create_tree_set_attribute(file_name):
    """
    Create a supply chain tree and set attributes.

    Parameters:
        file_name (str): Path to the tree file.

    Returns:
        tuple[dict, str]: Dictionary of Node instances and the root node name.
    """
    width_tracker = defaultdict(int)
    root_node_name = ""

    # Read the tree file
    rows = read_tree_file(file_name)
    nodes = {row["child_node_name"]: Node(row["child_node_name"]) for row in rows}

    for row in rows:
        if row["Parent_node"] == "root":
            root_node_name = row["Child_node"]
            root = nodes[root_node_name]
            root.width += 4
        else:
            parent = nodes[row["Parent_node"]]
            child = nodes[row["Child_node"]]
            parent.add_child(child)
            child.set_attributes(row)

    return nodes, root_node_name






# ******************************
# Evaluation process
# ******************************

def set_price_leaf2root(node, root_node_outbound, val):

    #print("node.name ", node.name)
    root_price = 0

    pb = 0
    pb = node.price_sales_shipped  # pb : Price_Base

    # set value on shipping price
    node.cs_price_sales_shipped = val

    print("def set_price_leaf2root", node.name, node.cs_price_sales_shipped )

    node.show_sum_cs()



    # cs : Cost_Stracrure
    node.cs_cost_total = val * node.cost_total / pb
    node.cs_profit = val * node.profit / pb
    node.cs_marketing_promotion = val * node.marketing_promotion / pb
    node.cs_sales_admin_cost = val * node.sales_admin_cost / pb
    node.cs_SGA_total = val * node.SGA_total / pb
    node.cs_custom_tax = val * node.custom_tax / pb
    node.cs_tax_portion = val * node.tax_portion / pb
    node.cs_logistics_costs = val * node.logistics_costs / pb
    node.cs_warehouse_cost = val * node.warehouse_cost / pb

    # direct shipping price that is,  like a FOB at port
    node.cs_direct_materials_costs = val * node.direct_materials_costs / pb

    node.cs_purchase_total_cost = val * node.purchase_total_cost / pb
    node.cs_prod_indirect_labor = val * node.prod_indirect_labor / pb
    node.cs_prod_indirect_others = val * node.prod_indirect_others / pb
    node.cs_direct_labor_costs = val * node.direct_labor_costs / pb
    node.cs_depreciation_others = val * node.depreciation_others / pb
    node.cs_manufacturing_overhead = val * node.manufacturing_overhead / pb

    #print("probe")
    #node.show_sum_cs()

    #print("node.cs_direct_materials_costs", node.name, node.cs_direct_materials_costs)
    #print("root_node_outbound.name", root_node_outbound.name)


    if node.name == root_node_outbound.name:
    #if node == root_node_outbound:

        node.cs_profit_accume = node.cs_profit # profit_accume�̏����Z�b�g

        root_price = node.cs_price_sales_shipped
        # root_price = node.cs_direct_materials_costs

        pass

    else:

        root_price = set_price_leaf2root(
            node.parent, root_node_outbound, node.cs_direct_materials_costs
        )

    return root_price




# 1st val is "root_price"
# ���̔��l=val���A��̎d����l=pb Price_Base portion�ɂȂ�B
def set_value_chain_outbound(val, node):


    # root_node��pass���āA�q������start


    # �͂��߂́Aroot_node�Ȃ̂�node.children�͑��݂���
    for child in node.children:

        #print("set_value_chain_outbound child.name ", child.name)
        # root_price = 0

        pb = 0
        pb = child.direct_materials_costs  # pb : Price_Base portion

        print("child.name", child.name)
        print("pb = child.direct_materials_costs",child.direct_materials_costs)

        # pb = child.price_sales_shipped # pb : Price_Base portion

        # direct shipping price that is,  like a FOB at port

        child.cs_direct_materials_costs = val

        # set value on shipping price
        child.cs_price_sales_shipped = val * child.price_sales_shipped / pb
        #print("def set_value_chain_outbound", child.name, child.cs_price_sales_shipped )
        child.show_sum_cs()



        val_child = child.cs_price_sales_shipped

        # cs : Cost_Stracrure
        child.cs_cost_total = val * child.cost_total / pb

        child.cs_profit = val * child.profit / pb

        # root2leaf�܂�profit_accume
        child.cs_profit_accume += node.cs_profit

        child.cs_marketing_promotion = val * child.marketing_promotion / pb
        child.cs_sales_admin_cost = val * child.sales_admin_cost / pb
        child.cs_SGA_total = val * child.SGA_total / pb
        child.cs_custom_tax = val * child.custom_tax / pb
        child.cs_tax_portion = val * child.tax_portion / pb
        child.cs_logistics_costs = val * child.logistics_costs / pb
        child.cs_warehouse_cost = val * child.warehouse_cost / pb

        ## direct shipping price that is,  like a FOB at port
        # node.cs_direct_materials_costs = val * node.direct_materials_costs / pb

        child.cs_purchase_total_cost = val * child.purchase_total_cost / pb
        child.cs_prod_indirect_labor = val * child.prod_indirect_labor / pb
        child.cs_prod_indirect_others = val * child.prod_indirect_others / pb
        child.cs_direct_labor_costs = val * child.direct_labor_costs / pb
        child.cs_depreciation_others = val * child.depreciation_others / pb
        child.cs_manufacturing_overhead = val * child.manufacturing_overhead / pb

        #print("probe")
        #child.show_sum_cs()


        print(
            "node.cs_direct_materials_costs",
            child.name,
            child.cs_direct_materials_costs,
        )
        # print("root_node_outbound.name", root_node_outbound.name )

        # to be rewritten@240803

        if child.children == []:  # leaf_node�Ȃ�I��

            pass

        else:  # ������������

            set_value_chain_outbound(val_child, child)

    # return


# **************************************
# call from gui.app
# **************************************




#@ STOP
#def eval_supply_chain_cost(node, context):
#    """
#    Recursively evaluates the cost of the entire supply chain.
#    
#    Parameters:
#        node (Node): The node currently being evaluated.
#        context (object): An object holding the total cost values (e.g., an instance of the GUI class).
#    """
#    # Count the number of lots for each node
#    node.set_lot_counts()
#
#    # Perform cost evaluation
#    total_revenue, total_profit = node.EvalPlanSIP_cost()
#
#    # Add the evaluation results to the context
#    context.total_revenue += total_revenue
#    context.total_profit += total_profit
#
#    # Recursively evaluate for child nodes
#    for child in node.children:
#        eval_supply_chain_cost(child, context)


# ******************************
# PSI evaluation on tree
# ******************************
# ******************************
# PSI evaluation on tree
# ******************************

def eval_supply_chain_cost(node, total_revenue=0, total_profit=0):
    """
    Recursively evaluates the cost of the supply chain for a given node.
    
    Parameters:
        node (Node): The root node to start the evaluation.
        total_revenue (float): Accumulated total revenue (default 0).
        total_profit (float): Accumulated total profit (default 0).

    Returns:
        Tuple[float, float]: Accumulated total revenue and total profit.
    """
    # Count the number of lots for the node
    node.set_lot_counts()

    # Evaluate the current node's costs
    node.revenue, node.profit = node.EvalPlanSIP_cost()



    # Accumulate the revenue and profit
    total_revenue += node.revenue
    total_profit  += node.profit

    # Recursively evaluate child nodes
    for child in node.children:
        total_revenue, total_profit = eval_supply_chain_cost(
            child, total_revenue, total_profit
        )

    return total_revenue, total_profit









# *****************
# network graph "node" "edge" process
# *****************




def make_edge_weight_capacity(node, child):
    # Calculate stock cost and customs tariff
    child.EvalPlanSIP_cost()

    #@ STOP
    #stock_cost = sum(child.WH_cost[1:])

    customs_tariff = child.customs_tariff_rate * child.cs_direct_materials_costs

    # Determine weight (logistics cost + tax + storage cost)
    cost_portion = 0.5
    weight4nx = max(0, child.cs_cost_total + (customs_tariff * cost_portion))

    # Calculate capacity (3 times the average weekly demand)
    demand_lots = sum(len(node.psi4demand[w][0]) for w in range(53 * node.plan_range))
    ave_demand_lots = demand_lots / (53 * node.plan_range)
    capacity4nx = 3 * ave_demand_lots

    # Add tariff to leaf nodes
    def add_tariff_on_leaf(node, customs_tariff):
        if not node.children:
            node.tariff_on_price += customs_tariff * cost_portion
        else:
            for child in node.children:
                add_tariff_on_leaf(child, customs_tariff)

    add_tariff_on_leaf(node, customs_tariff)

    # Logging for debugging (optional)
    print(f"child.name: {child.name}")
    print(f"weight4nx: {weight4nx}, capacity4nx: {capacity4nx}")

    return weight4nx, capacity4nx




def make_edge_weight_capacity_OLD(node, child):
    # Weight (�d��)
    #    - `weight`�́Aedge�Œ�`���ꂽ2�̃m�[�h�Ԃ̈ړ��R�X�g��\���B
    #       ������A�֐ŁA�ۊǃR�X�g�Ȃǂ̍��v���z�ɑΉ�����B
    #    - �Ⴆ�΁A������p�������ꍇ�A�Ή�����G�b�W��`weight`�͍����Ȃ�B
    #     �ŒZ�o�H�A���S���Y��(�_�C�N�X�g���@)��K�p����ƓK�؂Ȍo�H��I������B
    #
    #    self.demand�ɃZ�b�g?
    #

    # *********************
    # add_edge_parameter_set_weight_capacity()
    # add_edge()�̑O����
    # *********************
    # capacity
    # - `capacity`�́A�G�b�W�Œ�`���ꂽ2�̃m�[�h�Ԃɂ�������ԓ�����̈ړ���
    #   �̐����\���܂��B
    # - �T�v���C�`�F�[���̏ꍇ�A�ȉ��̃A�v���P�[�V��������������l������
    #   �l�b�N�����ƂȂ�ŏ��l��ݒ肷��B
    #     - ���ԓ��̃m�[�h�ԕ����̗e�ʂ̏���l
    #     - �ʊւ̊��ԓ������ʂ̏���l
    #     - �ۊǑq�ɂ̏���l
    #     - �o�ɁE�o�׍�Ƃ̊��ԓ������ʂ̏���l


    # *****************************************************
    # �݌ɕۊǃR�X�g�̎Z��̂��߂�eval�𗬂�
    # �q�m�[�h child.
    # *****************************************************
    stock_cost = 0

    #@ �v�m�F
    #@241231 �R�R�͐V����cost_table�ŕ]������
    child.EvalPlanSIP_cost()

    stock_cost = child.eval_WH_cost = sum(child.WH_cost[1:])

    customs_tariff = 0

    #@241231 �֐ŗ� X �d����P���Ƃ���
    customs_tariff = child.customs_tariff_rate * child.cs_direct_materials_costs

    print("child.name", child.name)
    print("child.customs_tariff_rate", child.customs_tariff_rate)
    print("child.cs_direct_materials_costs", child.cs_direct_materials_costs)
    print("customs_tariff", customs_tariff)

    print("self.cs_price_sales_shipped", node.cs_price_sales_shipped)
    print("self.cs_cost_total", node.cs_cost_total)
    print("self.cs_profit", node.cs_profit)


    #�֐ŃR�X�g�̋z�����@ 1
    # 1. ���v���ێ����A�R�X�g�Ɖ��i�ɏ�悹����B
    # 2. ���i���ێ����A�R�X�g�ɏ�悹���A���v�����B

    #    self.cs_price_sales_shipped # revenue
    #    self.cs_cost_total          # cost
    #    self.cs_profit              # profit


    #@ OLD STOP
    # �֐ŗ� X �P��
    #customs_tariff = child.customs_tariff_rate * child.REVENUE_RATIO



    weight4nx = 0


    # �����R�X�g
    # + TAX customs_tariff
    # + �݌ɕۊǃR�X�g
    # weight4nx = child.Distriburion_Cost + customs_tariff + stock_cost


    #@241231 ����:�֐ł̑���50%�𗘉v�팸����
    cost_portion = 0.5  # price_portion = 0.5 is following

    #@ RUN 
    weight4nx = child.cs_cost_total + (customs_tariff * cost_portion)



    #@ STOP
    #weight4nx = child.cs_profit_accume - (customs_tariff * cost_portion)
    #weight4nx =100*2 - child.cs_profit_accume + (customs_tariff *cost_portion)

    #print("child.cs_profit_accume", child.cs_profit_accume)

    print("child.cs_cost_total", child.cs_cost_total)

    print("customs_tariff", customs_tariff)
    print("cost_portion", cost_portion)
    print("weight4nx", weight4nx)



    if weight4nx < 0:
        weight4nx = 0


    # �o�׃R�X�g��PO_cost�Ɋ܂܂�Ă���
    ## �o�׃R�X�g
    # + xxxx

    #print("child.Distriburion_Cost", child.Distriburion_Cost)
    #print("+ TAX customs_tariff", customs_tariff)
    #print("+ stock_cost", stock_cost)
    #print("weight4nx", weight4nx)

    # ******************************
    # capacity4nx = 3 * average demand lots # ave weekly demand ��3�{��capa
    # ******************************
    capacity4nx = 0

    # ******************************
    # average demand lots
    # ******************************
    demand_lots = 0
    ave_demand_lots = 0

    for w in range(53 * node.plan_range):
        demand_lots += len(node.psi4demand[w][0])

    ave_demand_lots = demand_lots / (53 * node.plan_range)

    #@241231 ����:�֐ł̑���50%�����i�����ɂ����v�Ȑ���̉��i�e�͐�=1�Ƃ���
    # on the demand curve,
    # assume a price elasticity of demand of 1 due to price increase.

    #    self.cs_price_sales_shipped # revenue

    # demand_on_curve ���v�Ȑ���̎��v
    # customs_tariff*0.5 �֐ł�50%
    # self.cs_price_sales_shipped ����

    # ���i�e�͐��ɂ����v�ω�
    # customs_tariff*0.5 / self.cs_price_sales_shipped ���i������
    # self.price_elasticity
    # 0.0: demand "stay" like a medecine
    # 1.0: demand_decrease = price_increse * 1
    # 2.0: demand_decrease = price_increse * 2

    #@241231 MEMO demand_curve
    # �{���Ademand�͉��i�㏸���ɖ��[�s��leaf_node�ōi���邪�A
    # �����ł́A�ʊ֎��̒���node��capacity��demand���i�邱�Ƃœ��l�̌��ʂƂ���

    # ���[���i�ł͂Ȃ��̂ŁA�֐łɂ�鉿�i���������قȂ�?
    # customs_tariff:�֐ő�����cost��ޔ����Ă����Aself.customs_tariff
    # leaf_node�̖��[���i��demand_on_curve = ���i������ * node.price_elasticity


    # (customs_tariff * 0.5) ��lead_node��node.tariff_on_price��add

    def add_tariff_on_leaf(node, customs_tariff):

        price_portion = 0.5 # cost_portion = 0.5 is previously defined

        if node.children == []:  # leaf_node
            node.tariff_on_price += customs_tariff * price_portion # 0.5
        else:
            for child in node.children:
                add_tariff_on_leaf(child, customs_tariff)

    add_tariff_on_leaf(node, customs_tariff)

    #@ STOP
    #demand_on_curve = 3 * ave_demand_lots * (1-(customs_tariff*0.5 / node.cs_price_sales_shipped) * node.price_elasticity )
    #
    #capacity4nx = demand_on_curve       # 


    #@ STOP RUN
    capacity4nx = 3 * ave_demand_lots  # N * ave weekly demand

    print("weight4nx", weight4nx)
    print("capacity4nx", capacity4nx)

    return weight4nx, capacity4nx  # �R�R��float�̂܂ܖ߂�








def G_add_edge_from_tree(node, G):

    if node.children == []:  # leaf_node�𔻒�

        # ******************************
        # capacity4nx = average demand lots # ave weekly demand �����̂܂�set
        # ******************************
        capacity4nx = 0
        demand_lots = 0
        ave_demand_lots = 0

        for w in range(53 * node.plan_range):
            demand_lots += len(node.psi4demand[w][0])

        ave_demand_lots = demand_lots / (53 * node.plan_range)

        capacity4nx = ave_demand_lots  # N * ave weekly demand


        # ******************************
        # edge connecting leaf_node and "sales_office" �ڑ�
        # ******************************

        #@ RUN X1
        capacity4nx_int = round(capacity4nx) + 1

        #@ STOP
        # float2int X100
        #capacity4nx_int = float2int(capacity4nx)

        G.add_edge(node.name, "sales_office",
                 weight=0,
                 #capacity=capacity4nx_int
                 capacity=2000
        )

        print(
            "G.add_edge(node.name, office",
            node.name,
            "sales_office",
            "weight = 0, capacity =",
            capacity4nx,
        )

        # pass

    else:

        for child in node.children:

            # *****************************
            # make_edge_weight_capacity
            # *****************************
            weight4nx, capacity4nx = make_edge_weight_capacity(node, child)

            # float2int
            weight4nx_int = float2int(weight4nx)

            #@ RUN X1
            capacity4nx_int = round(capacity4nx) + 1

            #@ STOP
            # float2int X100
            #capacity4nx_int = float2int(capacity4nx)

            child.nx_weight = weight4nx_int
            child.nx_capacity = capacity4nx_int

            # ******************************
            # edge connecting self.node & child.node
            # ******************************

            G.add_edge(
                node.name, child.name, 
                weight=weight4nx_int,

                #capacity=capacity4nx_int
                capacity=2000

            )

            print(
                "G.add_edge(node.name, child.name",
                node.name,
                child.name,
                "weight =",
                weight4nx_int,
                "capacity =",
                capacity4nx_int,
            )

            G_add_edge_from_tree(child, G)





def Gsp_add_edge_sc2nx_inbound(node, Gsp):

    if node.children == []:  # leaf_node�𔻒�

        # ******************************
        # capacity4nx = average demand lots # ave weekly demand �����̂܂�set
        # ******************************
        capacity4nx = 0
        demand_lots = 0
        ave_demand_lots = 0

        for w in range(53 * node.plan_range):
            demand_lots += len(node.psi4demand[w][0])

        ave_demand_lots = demand_lots / (53 * node.plan_range)

        capacity4nx = ave_demand_lots  # N * ave weekly demand

        # ******************************
        # edge connecting leaf_node and "sales_office" �ڑ�
        # ******************************

        # float2int
        capacity4nx_int = float2int(capacity4nx)

        Gsp.add_edge( "procurement_office", node.name,
                 weight=0,
                 capacity = 2000 # 240906 TEST # capacity4nx_int * 1 # N�{
                 #capacity=capacity4nx_int * 1 # N�{
        )

        # pass

    else:

        for child in node.children:

            # *****************************
            # make_edge_weight_capacity
            # *****************************
            weight4nx, capacity4nx = make_edge_weight_capacity(node, child)

            # float2int
            weight4nx_int = float2int(weight4nx)
            capacity4nx_int = float2int(capacity4nx)

            #@240906 TEST 
            capacity4nx_int = 2000

            child.nx_weight = weight4nx_int
            child.nx_capacity = capacity4nx_int

            # ******************************
            # edge connecting self.node & child.node
            # ******************************

            Gsp.add_edge(
                child.name, node.name, 
                weight=weight4nx_int,

                capacity=capacity4nx_int
            )

            Gsp_add_edge_sc2nx_inbound(child, Gsp)





def Gdm_add_edge_sc2nx_outbound(node, Gdm):

    if node.children == []:  # leaf_node�𔻒�

        # ******************************
        # capacity4nx = average demand lots # ave weekly demand �����̂܂�set
        # ******************************
        capacity4nx = 0
        demand_lots = 0
        ave_demand_lots = 0

        for w in range(53 * node.plan_range):
            demand_lots += len(node.psi4demand[w][0])



        ave_demand_lots = demand_lots / (53 * node.plan_range)

        #@ STOP
        #capacity4nx = ave_demand_lots  # N * ave weekly demand

        tariff_portion = node.tariff_on_price / node.cs_price_sales_shipped

        demand_on_curve = 3 * ave_demand_lots * (1- tariff_portion) * node.price_elasticity 


        print("node.name", node.name)

        print("node.tariff_on_price", node.tariff_on_price)
        print("node.cs_price_sales_shipped", node.cs_price_sales_shipped)
        print("tariff_portion", tariff_portion)

        print("ave_demand_lots", ave_demand_lots)
        print("node.price_elasticity", node.price_elasticity)
        print("demand_on_curve", demand_on_curve)



        #demand_on_curve = 3 * ave_demand_lots * (1-(customs_tariff*0.5 / node.cs_price_sales_shipped) * node.price_elasticity )

        capacity4nx = demand_on_curve       # 


        print("capacity4nx", capacity4nx)



        # ******************************
        # edge connecting leaf_node and "sales_office" �ڑ�
        # ******************************

        # float2int
        capacity4nx_int = float2int(capacity4nx)

        # set PROFIT 2 WEIGHT

        Gdm.add_edge(node.name, "sales_office",
                 weight=0,
                 capacity=capacity4nx_int * 1 # N�{
        )

        print(
            "Gdm.add_edge(node.name, office",
            node.name,
            "sales_office",
            "weight = 0, capacity =",
            capacity4nx,
        )

        # pass

    else:

        for child in node.children:

            # *****************************
            # make_edge_weight_capacity
            # *****************************
            weight4nx, capacity4nx = make_edge_weight_capacity(node, child)

            # float2int
            weight4nx_int = float2int(weight4nx)
            capacity4nx_int = float2int(capacity4nx)

            child.nx_weight = weight4nx_int
            child.nx_capacity = capacity4nx_int

            # ******************************
            # edge connecting self.node & child.node
            # ******************************

            Gdm.add_edge(
                node.name, child.name, 
                weight=weight4nx_int,

                capacity=capacity4nx_int
            )

            print(
                "Gdm.add_edge(node.name, child.name",
                node.name, child.name,
                "weight =", weight4nx_int,
                "capacity =", capacity4nx_int
            )

            Gdm_add_edge_sc2nx_outbound(child, Gdm)






def make_edge_weight(node, child):


#NetworkX�ł́A�G�b�W�̏d�݁iweight�j���傫���ꍇ�A���̃G�b�W�̗��p�D��x�́A�A���S���Y����ړI�ɂ���ĈقȂ�

    # Weight (�d��)
    #    - `weight`��edge�Œ�`���ꂽ2�m�[�h�ԂŔ�������profit(rev-cost)�ŕ\��
    #       cost=������A�֐ŁA�ۊǃR�X�g�Ȃǂ̍��v���z�ɑΉ�����B
    #    - �Ⴆ�΁A������p�������ꍇ�A�Ή�����G�b�W��`weight`�͒Ⴍ�Ȃ�B
    #     �ŒZ�o�H�A���S���Y��(�_�C�N�X�g���@)��K�p����ƓK�؂Ȍo�H��I������B

#�ŒZ�o�H�A���S���Y���i��FDijkstra�fs algorithm�j�ł́A�G�b�W�̏d�݂��傫���قǁA���̃G�b�W��ʂ�o�H�̃R�X�g�������Ȃ邽�߁A�D��x�͉�����

#�ő�t���[���Ȃǂ̑��̃A���S���Y���ł́A�G�b�W�̏d�݂��傫���قǁA���̃G�b�W��ʂ�t���[�������Ȃ邽�߁A�D��x���オ�邱�Ƃ�����
#��̓I�ȏ󋵂�g�p����A���S���Y���ɂ���ĈقȂ邽�߁A
#�ړI�ɉ����ēK�؂ȃA���S���Y����I�����邱�Ƃ��d�v

# �ő�t���[���iMaximum Flow Problem�j
# �t�H�[�h�E�t�@���J�[�\���@ (Ford-Fulkerson Algorithm)
#�t�H�[�h�E�t�@���J�[�\���@�́A�l�b�g���[�N���̃\�[�X�i�n�_�j����V���N�i�I�_�j�܂ł̍ő�t���[��������A���S���Y��
#���̃A���S���Y���ł́A�G�b�W�̏d�݁i�e�ʁj���傫���قǁA���̃G�b�W��ʂ�t���[�������Ȃ邽�߁A�D��x���オ��܂��B


#@240831 
#    # *****************************************************
#    # �݌ɕۊǃR�X�g�̎Z��̂��߂�eval�𗬂�
#    # �q�m�[�h child.
#    # *****************************************************
#
#    stock_cost = 0
#
#    child.EvalPlanSIP()
#
#    stock_cost = child.eval_WH_cost = sum(child.WH_cost[1:])
#
#    customs_tariff = 0
#    customs_tariff = child.customs_tariff_rate * child.REVENUE_RATIO  # �֐ŗ� X �P��
#
#    # �����R�X�g
#    # + TAX customs_tariff
#    # + �݌ɕۊǃR�X�g
#    # weight4nx = child.Distriburion_Cost + customs_tariff + stock_cost


    # priority is "profit"

    weight4nx = 0

    weight4nx = child.cs_profit_accume

    return weight4nx






#@240830 �R�����C��
# 1.capacity�̌v�Z�́Asupply side�Ő��i���b�g�P�ʂ̓��ꂵ��root_capa * N�{
# 2.��node=>�enode�̊֌W��` G.add_edge(self.node, parent.node)

def G_add_edge_from_inbound_tree(node, supplyers_capacity, G):

    if node.children == []:  # leaf_node�𔻒�

        # ******************************
        # capacity4nx = average demand lots # ave weekly demand *N�{��set
        # ******************************
        capacity4nx = 0

        # 
        # ******************************
        #demand_lots = 0
        #ave_demand_lots = 0
        #
        #for w in range(53 * node.plan_range):
        #    demand_lots += len(node.psi4demand[w][0])
        #
        #ave_demand_lots = demand_lots / (53 * node.plan_range)
        #
        #capacity4nx = ave_demand_lots * 5  # N * ave weekly demand
        #
        # ******************************

        # supplyers_capacity�́Aroot_node=mother plant��capacity
        # ���[suppliers�́A���ς�5�{��capa
        capacity4nx = supplyers_capacity * 5  # N * ave weekly demand


        # float2int
        capacity4nx_int = float2int(capacity4nx)

        # ******************************
        # edge connecting leaf_node and "procurement_office" �ڑ�
        # ******************************

        G.add_edge("procurement_office", node.name, weight=0, capacity=2000)

        #G.add_edge("procurement_office", node.name, weight=0, capacity=capacity4nx_int)

        print(
            "G.add_edge(node.name, office",
            node.name,
            "sales_office",
            "weight = 0, capacity =",
            capacity4nx,
        )

        # pass

    else:

        for child in node.children:


            # supplyers_capacity�́Aroot_node=mother plant��capacity
            # ����suppliers�́A���ς�3�{��capa
            capacity4nx = supplyers_capacity * 3  # N * ave weekly demand


            # *****************************
            # set_edge_weight
            # *****************************
            weight4nx = make_edge_weight(node, child)

            ## *****************************
            ## make_edge_weight_capacity
            ## *****************************
            #weight4nx, capacity4nx = make_edge_weight_capacity(node, child)



            # float2int
            weight4nx_int = float2int(weight4nx)
            capacity4nx_int = float2int(capacity4nx)

            child.nx_weight = weight4nx_int
            child.nx_capacity = capacity4nx_int

            # ******************************
            # edge connecting from child.node to self.node as INBOUND
            # ******************************
            #G.add_edge(
            #    child.name, node.name, 
            #    weight=weight4nx_int, capacity=capacity4nx_int
            #)

            G.add_edge(
                child.name, node.name, 
                weight=weight4nx_int, capacity=2000
            )

            #print(
            #    "G.add_edge(child.name, node.name ",
            #    child.name,
            #    node.name,
            #    "weight =",
            #    weight4nx_int,
            #    "capacity =", 
            #    capacity4nx_int,
            #)

            G_add_edge_from_inbound_tree(child, supplyers_capacity, G)






    # *********************
    # OUT tree��T������G.add_node����������
    # node_name��G�ɃZ�b�g (X,Y)��free�ȏ�ԁA(X,Y)��setting�͌㏈��
    # *********************
def G_add_nodes_from_tree(node, G):


    G.add_node(node.name, demand=0)
    #G.add_node(node.name, demand=node.nx_demand) #demand�͋��������NOT set!!

    print("G.add_node", node.name, "demand =", node.nx_demand)

    if node.children == []:  # leaf_node�̏ꍇ�Atotal_demand�ɉ��Z

        pass

    else:

        for child in node.children:

            G_add_nodes_from_tree(child, G)



    # *********************
    # IN tree��T������G.add_node����������B�������Aroot_node_inbound��skip
    # node_name��G�ɃZ�b�g (X,Y)��free�ȏ�ԁA(X,Y)��setting�͌㏈��
    # *********************
def G_add_nodes_from_tree_skip_root(node, root_node_name_in, G):

    #@240901STOP
    #if node.name == root_node_name_in:
    #
    #    pass
    #
    #else:
    #
    #    G.add_node(node.name, demand=0)
    #    print("G.add_node", node.name, "demand = 0")

    G.add_node(node.name, demand=0)
    print("G.add_node", node.name, "demand = 0")

    if node.children == []:  # leaf_node�̏ꍇ

        pass

    else:

        for child in node.children:

            G_add_nodes_from_tree_skip_root(child, root_node_name_in, G)
        





# *****************
# demand, weight and scaling FLOAT to INT
# *****************
def float2int(value):

    scale_factor = 100
    scaled_demand = value * scale_factor

    # �l�̌ܓ�
    rounded_demand = round(scaled_demand)
    # print(f"�l�̌ܓ�: {rounded_demand}")

    ## �؂�̂�
    # floored_demand = math.floor(scaled_demand)
    # print(f"�؂�̂�: {floored_demand}")

    ## �؂�グ
    # ceiled_demand = math.ceil(scaled_demand)
    # print(f"�؂�グ: {ceiled_demand}")

    return rounded_demand



# *********************
# ���[�s��A�ŏI����̔̔��`���l����demand = leaf_node_demand
# tree��leaf nodes��T������"weekly average base"��total_demand���W�v
# *********************
def set_leaf_demand(node, total_demand):

    if node.children == []:  # leaf_node�̏ꍇ�Atotal_demand�ɉ��Z

        # ******************************
        # average demand lots
        # ******************************
        demand_lots = 0
        ave_demand_lots = 0
        ave_demand_lots_int = 0

        for w in range(53 * node.plan_range):
            demand_lots += len(node.psi4demand[w][0])

        ave_demand_lots = demand_lots / (53 * node.plan_range)

        # float2int
        ave_demand_lots_int = float2int(ave_demand_lots)


        # **** networkX demand *********
        # set demand on leaf_node    
        # weekly average demand by lot
        # ******************************
        node.nx_demand = ave_demand_lots_int


        total_demand += ave_demand_lots_int

    else:

        for child in node.children:

            # "�s��" GOing on the way

            total_demand = set_leaf_demand(child, total_demand)


            # "�A��" RETURNing on the way BACK
            node.nx_demand = child.nx_demand  # set "middle_node" demand


    return total_demand





# ***************************
# make network with NetworkX
# show network with plotly
# ***************************



def calc_put_office_position(pos_office, office_name):
    x_values = [pos_office[key][0] for key in pos_office]
    max_x = max(x_values)
    y_values = [pos_office[key][1] for key in pos_office]
    max_y = max(y_values)
    pos_office[office_name] = (max_x + 1, max_y + 1)
    return pos_office

def generate_positions(node, pos, depth=0, y_offset=0, leaf_y_positions=None):
    if not node.children:
        pos[node.name] = (depth, leaf_y_positions.pop(0))
    else:
        child_y_positions = []
        for child in node.children:
            generate_positions(child, pos, depth + 1, y_offset, leaf_y_positions)
            child_y_positions.append(pos[child.name][1])
        pos[node.name] = (depth, sum(child_y_positions) / len(child_y_positions))  # �q�m�[�h��Y�����ϒl��e�m�[�h�ɐݒ�
    return pos

def count_leaf_nodes(node):
    if not node.children:
        return 1
    return sum(count_leaf_nodes(child) for child in node.children)

def get_leaf_y_positions(node, y_positions=None):
    if y_positions is None:
        y_positions = []
    if not node.children:
        y_positions.append(len(y_positions))
    else:
        for child in node.children:
            get_leaf_y_positions(child, y_positions)
    return y_positions

def tune_hammock(pos_E2E, nodes_outbound, nodes_inbound):
    # Compare 'procurement_office' and 'sales_office' Y values and choose the larger one
    procurement_office_y = pos_E2E['procurement_office'][1]
    office_y = pos_E2E['sales_office'][1]

    max_y = max(procurement_office_y, office_y)
    
    pos_E2E['procurement_office'] = (pos_E2E['procurement_office'][0], max_y)
    pos_E2E['sales_office'] = (pos_E2E['sales_office'][0], max_y)

    # Align 'FA_xxxx' and 'PL_xxxx' pairs and their children
    for key, value in pos_E2E.items():
        if key.startswith('MOM'):
            corresponding_key = 'DAD' + key[3:]
            if corresponding_key in pos_E2E:
                fa_y = value[1]
                pl_y = pos_E2E[corresponding_key][1]
                aligned_y = max(fa_y, pl_y)
                pos_E2E[key] = (value[0], aligned_y)
                pos_E2E[corresponding_key] = (pos_E2E[corresponding_key][0], aligned_y)


                offset_y = max( aligned_y - fa_y, aligned_y - pl_y )

                if aligned_y - fa_y == 0: # inbound�̍��������� outbound�𒲐�
                    
                    pool_node = nodes_outbound[corresponding_key]
                    adjust_child_positions(pool_node, pos_E2E, offset_y)

                else:

                    fassy_node = nodes_inbound[key]
                    adjust_child_positions(fassy_node, pos_E2E, offset_y)



                ## Adjust children nodes
                #adjust_child_positions(pos_E2E, key, aligned_y)
                #adjust_child_positions(pos_E2E, corresponding_key, aligned_y)

    return pos_E2E

#def adjust_child_positions(pos, parent_key, parent_y):
#    for key, value in pos.items():
#        if key != parent_key and pos[key][0] > pos[parent_key][0]:
#            pos[key] = (value[0], value[1] + (parent_y - pos[parent_key][1]))


def adjust_child_positions(node, pos, offset_y):
    if node.children == []:  # leaf_node�𔻒�
        pass
    else:
        for child in node.children:
            # y�̍����𒲐� 
            pos[child.name] = (pos[child.name][0], pos[child.name][1]+offset_y)
            adjust_child_positions(child, pos, offset_y)


def make_E2E_positions(root_node_outbound, root_node_inbound):
    out_leaf_count = count_leaf_nodes(root_node_outbound)
    in_leaf_count = count_leaf_nodes(root_node_inbound)

    print("out_leaf_count", out_leaf_count)
    print("in_leaf_count", in_leaf_count)

    out_leaf_y_positions = get_leaf_y_positions(root_node_outbound)
    in_leaf_y_positions = get_leaf_y_positions(root_node_inbound)

    pos_out = generate_positions(root_node_outbound, {}, leaf_y_positions=out_leaf_y_positions)
    pos_out = calc_put_office_position(pos_out, "sales_office")

    pos_in = generate_positions(root_node_inbound, {}, leaf_y_positions=in_leaf_y_positions)
    pos_in = calc_put_office_position(pos_in, "procurement_office")

    max_x = max(x for x, y in pos_in.values())
    pos_in_reverse = {node: (max_x - x, y) for node, (x, y) in pos_in.items()}
    pos_out_shifting = {node: (x + max_x, y) for node, (x, y) in pos_out.items()}

    merged_dict = pos_in_reverse.copy()
    for key, value in pos_out_shifting.items():
        if key in merged_dict:
            if key == root_node_outbound.name:
                merged_dict[key] = value if value[1] > merged_dict[key][1] else merged_dict[key]
            else:
                merged_dict[key] = value
        else:
            merged_dict[key] = value

    pos_E2E = merged_dict

    return pos_E2E











if __name__ == "__main__":

    # Example usage
    example_tree = {
        "root": ["child1", "child2"],
        "child1": ["child1_1", "child1_2"],
        "child2": ["child2_1"]
    }

    root_node = build_tree_from_dict(example_tree)
    root_node.print_tree()


