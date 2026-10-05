"""设备状态评价的最小贝叶斯网络。

网络结构：
    VoltageUnstable -> DeviceState
    TempHigh        -> DeviceState
    Age             -> DeviceState

    因素 -> 离散状态 -> CPT -> 给证据 -> 计算设备状态后验概率
"""

from __future__ import annotations

import sys

# 让 Windows 终端/PyCharm 运行输出尽量保持 UTF-8，
# 避免中文标签显示成乱码。即使当前对象没有 reconfigure，也不会报错。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ------------------------------------------------------------
# 兼容新旧版 pgmpy：
# 新版：DiscreteBayesianNetwork
# 旧版：BayesianNetwork
# ------------------------------------------------------------
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork


# TabularCPD：用表格形式描述“当前节点在父节点条件下的概率”。
from pgmpy.factors.discrete import TabularCPD

# VariableElimination：变量消元推理器，用于计算 P(查询变量 | 证据)。
from pgmpy.inference import VariableElimination


# 数字状态和中文含义的对应关系。
# pgmpy 本身可以只使用 0、1、2，这里另外准备字典，
# 这样打印结果时可以显示成“正常 / 预警 / 故障”等中文。
LABELS = {
    "VoltageUnstable": {
        0: "电压稳定",
        1: "电压不稳",
    },
    "TempHigh": {
        0: "温度正常",
        1: "温度过高",
    },
    "Age": {
        0: "新设备",
        1: "使用 3-8 年",
        2: "使用 8 年以上",
    },
    "DeviceState": {
        0: "正常",
        1: "预警",
        2: "故障",
    },
}


def build_device_network() -> BayesianNetwork:
    """建立设备状态网络，填写所有 CPT，并返回模型。"""

    # --------------------------------------------------------
    # 网络结构 DAG。
    #
    # 每一个小括号表示一条有向边：
    # ("父节点", "子节点")
    #
    # VoltageUnstable、TempHigh、Age 都是 DeviceState 的父节点。
    # 这三个风险因素共同影响设备状态。
    # --------------------------------------------------------
    model = BayesianNetwork(
        [
            ("VoltageUnstable", "DeviceState"),
            ("TempHigh", "DeviceState"),
            ("Age", "DeviceState"),
        ]
    )

    # --------------------------------------------------------
    # 根节点 VoltageUnstable 的 CPT。
    #
    # 状态：
    #   0 = 电压稳定
    #   1 = 电压不稳
    #
    # 先验概率：
    #   电压稳定 75%
    #   电压不稳 25%
    #
    # 每一列概率和为 1：
    #   0.75 + 0.25 = 1
    # --------------------------------------------------------
    cpd_voltage = TabularCPD(
        variable="VoltageUnstable",
        variable_card=2,
        values=[[0.75], [0.25]],
    )

    # --------------------------------------------------------
    # 根节点 TempHigh 的 CPT。
    #
    # 状态：
    #   0 = 温度正常
    #   1 = 温度过高
    #
    # 先验概率：
    #   温度正常 80%
    #   温度过高 20%
    #
    # 每一列概率和为 1：
    #   0.80 + 0.20 = 1
    # --------------------------------------------------------
    cpd_temp = TabularCPD(
        variable="TempHigh",
        variable_card=2,
        values=[[0.80], [0.20]],
    )

    # --------------------------------------------------------
    # 根节点 Age 的 CPT。
    #
    # 状态：
    #   0 = 新设备
    #   1 = 使用 3-8 年
    #   2 = 使用 8 年以上
    #
    # 先验概率：
    #   新设备          50%
    #   使用 3-8 年     30%
    #   使用 8 年以上   20%
    #
    # 每一列概率和为 1：
    #   0.50 + 0.30 + 0.20 = 1
    # --------------------------------------------------------
    cpd_age = TabularCPD(
        variable="Age",
        variable_card=3,
        values=[[0.50], [0.30], [0.20]],
    )

    # --------------------------------------------------------
    # DeviceState 的 CPT：
    #
    # P(DeviceState | VoltageUnstable, TempHigh, Age)
    #
    # 父节点顺序由 evidence 决定：
    #   evidence=["VoltageUnstable", "TempHigh", "Age"]
    #
    # 因此 12 列的顺序是：
    #   (V=0,T=0,A=0), (V=0,T=0,A=1), (V=0,T=0,A=2),
    #   (V=0,T=1,A=0), (V=0,T=1,A=1), (V=0,T=1,A=2),
    #   (V=1,T=0,A=0), (V=1,T=0,A=1), (V=1,T=0,A=2),
    #   (V=1,T=1,A=0), (V=1,T=1,A=1), (V=1,T=1,A=2)
    #
    # 三个父节点的状态数分别是 2、2、3，
    # 所以一共有：
    #   2 × 2 × 3 = 12 列。
    #
    # values 的每一行对应 DeviceState 的一个状态：
    #   第 1 行：DeviceState=0，正常
    #   第 2 行：DeviceState=1，预警
    #   第 3 行：DeviceState=2，故障
    #
    # 每一列三个概率相加都必须等于 1。
    # --------------------------------------------------------
    cpd_state = TabularCPD(
        variable="DeviceState",
        variable_card=3,
        values=[
            # DeviceState=0：正常
            # 12 个数字依次对应上面列出的 12 组 (V,T,A)。
            [0.94, 0.84, 0.62, 0.72, 0.55, 0.30,
             0.58, 0.38, 0.18, 0.25, 0.12, 0.04],

            # DeviceState=1：预警
            # 同样依次对应 12 组 (V,T,A)。
            [0.05, 0.13, 0.28, 0.22, 0.32, 0.45,
             0.30, 0.42, 0.47, 0.45, 0.43, 0.36],

            # DeviceState=2：故障
            # 同样依次对应 12 组 (V,T,A)。
            [0.01, 0.03, 0.10, 0.06, 0.13, 0.25,
             0.12, 0.20, 0.35, 0.30, 0.45, 0.60],
        ],

        # 三个父节点及顺序。
        evidence=["VoltageUnstable", "TempHigh", "Age"],

        # 三个父节点的状态数量，必须和 evidence 一一对应。
        evidence_card=[2, 2, 3],
    )

    # --------------------------------------------------------
    # 把 4 个 CPT 加入模型。
    # 三个根节点各有一张先验 CPT，DeviceState 有一张条件 CPT。
    # --------------------------------------------------------
    model.add_cpds(cpd_voltage, cpd_temp, cpd_age, cpd_state)

    # 检查模型是否完整、CPT 是否和节点匹配、概率是否合法。
    model.check_model()

    return model


def show_query(name: str, factor) -> None:
    """把 pgmpy 返回的概率分布打印成中英文都容易读的形式。"""

    print(f"\n{name}")

    # 查询结果 factor 里有：
    #   variables: 查询变量列表
    #   state_names: 状态名称
    #   values: 每个状态对应的概率
    variable = factor.variables[0]

    for state, probability in zip(factor.state_names[variable], factor.values):
        # 状态可能是 numpy 整数，因此先转成 int 再查中文标签。
        label = LABELS.get(variable, {}).get(int(state), str(state))
        print(f"  {state} ({label}): {probability:.6f}")


def main() -> None:
    """运行整个设备状态评价 Demo。"""

    # 先建立模型，再创建变量消元推理器。
    model = build_device_network()
    infer = VariableElimination(model)

    print("设备状态网络构建成功。")
    print("节点：", list(model.nodes()))
    print("有向边：", list(model.edges()))

    # --------------------------------------------------------
    # 查询 1：没有任何证据时的设备状态先验分布。
    #
    # 因为没有给 VoltageUnstable、TempHigh、Age 任何证据，
    # 推断会根据三个因素的先验概率对所有风险组合加权平均。
    # --------------------------------------------------------
    prior = infer.query(
        variables=["DeviceState"],
        show_progress=False,
    )
    show_query("P(DeviceState)：没有任何证据", prior)

    # --------------------------------------------------------
    # 查询 2：已知电压不稳、温度过高，但不知道设备年龄。
    #
    # 证据：
    #   VoltageUnstable=1
    #   TempHigh=1
    #
    # Age 没有给证据，所以会按 Age 的先验概率加权：
    #   Age=0 概率 0.5
    #   Age=1 概率 0.3
    #   Age=2 概率 0.2
    # --------------------------------------------------------
    evidence_1 = {
        "VoltageUnstable": 1,
        "TempHigh": 1,
    }
    posterior_1 = infer.query(
        variables=["DeviceState"],
        evidence=evidence_1,
        show_progress=False,
    )
    show_query(
        "P(DeviceState | 电压不稳, 温度过高)",
        posterior_1,
    )

    # --------------------------------------------------------
    # 查询 3：三个风险因素都已知，而且年龄是 8 年以上。
    #
    # 这时不需要再对 Age 加权，直接读取 CPT 中
    # (V=1,T=1,A=2) 对应的那一列：
    #   正常 0.04
    #   预警 0.36
    #   故障 0.60
    # --------------------------------------------------------
    evidence_2 = {
        "VoltageUnstable": 1,
        "TempHigh": 1,
        "Age": 2,
    }
    posterior_2 = infer.query(
        variables=["DeviceState"],
        evidence=evidence_2,
        show_progress=False,
    )
    show_query(
        "P(DeviceState | 电压不稳, 温度过高, 使用 8 年以上)",
        posterior_2,
    )

    # --------------------------------------------------------
    # 对照查询：电压不稳、使用 8 年以上，但温度正常。
    #
    # 这组证据对应 CPT 中的 (V=1,T=0,A=2) 列：
    #   正常 0.18
    #   预警 0.47
    #   故障 0.35
    #
    # 和查询 3 对比：
    # 去掉“温度过高”这个风险证据后，
    # 故障后验概率从 0.60 降到 0.35。
    # --------------------------------------------------------
    counterfactual = infer.query(
        variables=["DeviceState"],
        evidence={
            "VoltageUnstable": 1,
            "TempHigh": 0,
            "Age": 2,
        },
        show_progress=False,
    )
    show_query(
        "对照：P(DeviceState | 电压不稳, 温度正常, 使用 8 年以上)",
        counterfactual,
    )

if __name__ == "__main__":
    main()