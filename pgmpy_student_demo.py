# ============================================================
# Student 学生成绩贝叶斯网络示例
#
# 1. 建立 5 个节点的贝叶斯网络结构。
# 2. 为每个节点设置一个 CPD，也就是条件概率分布。
# 3. 把 CPD 加入模型。
# 4. 检查模型是否完整、合法。
# 5. 打印 Letter 节点以及所有节点的概率表。
#
# 注意：
# 没有设置 state_names，因此 pgmpy 默认使用数字状态：
# Intelligence：0=智力不高，1=智力高
# Difficulty  ：0=课程容易，1=课程难
# SAT         ：0=低分，1=高分
# Grade       ：0=A，1=B，2=C
# Letter      ：0=普通推荐信，1=强推荐信
# ============================================================


# ------------------------------------------------------------
# 兼容不同版本的 pgmpy：
# ------------------------------------------------------------
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork


# ------------------------------------------------------------
# TabularCPD：
# 用来创建“表格形式”的条件概率分布。
#
# CPD = Conditional Probability Distribution，条件概率分布。
# CPT = Conditional Probability Table，条件概率表。
#
# 在离散贝叶斯网络中，CPD 通常以 CPT 的形式表示。
# ------------------------------------------------------------
from pgmpy.factors.discrete import TabularCPD


# ------------------------------------------------------------
# VariableElimination：
# pgmpy 中的变量消元推理工具。
#
# 它以后可以用来计算类似：
# P(Intelligence | Grade=A)
#
# 但是这段代码虽然导入了它，还没有真正调用它，
# 所以当前代码只负责搭建和打印模型，不做后验推理。
# ------------------------------------------------------------
from pgmpy.inference import VariableElimination


# ------------------------------------------------------------
# 创建贝叶斯网络模型。
#
# 列表中的每一个小括号表示一条有向边：
# ("父节点", "子节点")
#
# 因此这里建立了以下结构：
# Intelligence -> SAT
# Intelligence -> Grade
# Difficulty   -> Grade
# Grade        -> Letter
# SAT          -> Letter
#
# 这个结构是一个 DAG，也就是有向无环图：
# - “有向”表示箭头有方向。
# - “无环”表示沿着箭头走，不能绕回原来的节点。
# ------------------------------------------------------------
model = BayesianNetwork([

    # Intelligence 直接影响 SAT：
    # 智力水平会影响 SAT 成绩。
    ("Intelligence", "SAT"),

    # Intelligence 直接影响 Grade：
    # 智力水平会影响课程成绩。
    ("Intelligence", "Grade"),

    # Difficulty 直接影响 Grade：
    # 课程难度也会影响课程成绩。
    ("Difficulty", "Grade"),

    # Grade 直接影响 Letter：
    # 课程成绩会影响推荐信质量。
    ("Grade", "Letter"),

    # SAT 直接影响 Letter：
    # SAT 成绩也会影响推荐信质量。
    ("SAT", "Letter"),
])


# ------------------------------------------------------------
# 建立 Intelligence 节点的 CPD。
#
# Intelligence 没有父节点，所以它是根节点。
# 根节点的 CPD 是一个普通的先验概率分布。
#
# variable：
# 当前 CPD 所属的变量名。
#
# variable_card=2：
# 当前变量有 2 个状态：
# 状态 0 = 智力不高
# 状态 1 = 智力高
#
# values：
# 每行对应一个状态，每列对应一种父节点组合。
# Intelligence 没有父节点，所以只有 1 列。
# ------------------------------------------------------------
cpd_i = TabularCPD(
    variable="Intelligence",
    variable_card=2,

    # 第 1 行对应 Intelligence=0：智力不高的概率是 0.7。
    # 第 2 行对应 Intelligence=1：智力高的概率是 0.3。
    # 两个概率相加：
    # 0.7 + 0.3 = 1
    values=[[0.7], [0.3]],
)


# ------------------------------------------------------------
# 建立 Difficulty 节点的 CPD。
#
# Difficulty 没有父节点，所以它也是根节点。
#
# 状态 0 = 课程容易
# 状态 1 = 课程难
# ------------------------------------------------------------
cpd_d = TabularCPD(
    variable="Difficulty",
    variable_card=2,

    # 第 1 行对应 Difficulty=0：课程容易的概率是 0.6。
    # 第 2 行对应 Difficulty=1：课程难的概率是 0.4。
    # 两个概率相加：
    # 0.6 + 0.4 = 1
    values=[[0.6], [0.4]],
)


# ------------------------------------------------------------
# 建立 SAT 节点的 CPD。
#
# SAT 有一个父节点：Intelligence。
#
# 这个 CPD 表示：
# P(SAT | Intelligence)
#
# 也就是：
# 在智力水平已知的情况下，SAT 成绩的概率分布。
#
# variable_card=2：
# SAT 有 2 个状态：
# SAT=0：低分
# SAT=1：高分
#
# evidence：
# 父节点列表。
# 这里只有一个父节点 Intelligence。
#
# evidence_card：
# 每个父节点分别有多少个状态。
# [2] 表示 Intelligence 有 2 个状态。
# ------------------------------------------------------------
cpd_sat = TabularCPD(
    variable="SAT",
    variable_card=2,

    # values 的每一行对应 SAT 的一个状态。
    # values 的每一列对应 Intelligence 的一种取值。
    #
    # 列顺序由 evidence=["Intelligence"] 决定：
    # 第 1 列：Intelligence=0，智力不高
    # 第 2 列：Intelligence=1，智力高
    values=[
        # 第 1 行对应 SAT=0，也就是 SAT 低分：
        # 智力不高时，低分概率是 0.95。
        # 智力高时，低分概率是 0.20。
        [0.95, 0.20],

        # 第 2 行对应 SAT=1，也就是 SAT 高分：
        # 智力不高时，高分概率是 0.05。
        # 智力高时，高分概率是 0.80。
        [0.05, 0.80],
    ],

    # SAT 的父节点是 Intelligence。
    evidence=["Intelligence"],

    # Intelligence 有 2 个状态。
    # evidence_card 的顺序必须和 evidence 一一对应。
    evidence_card=[2],
)


# ------------------------------------------------------------
# 建立 Grade 节点的 CPD。
#
# Grade 有两个父节点：
# 1. Intelligence
# 2. Difficulty
#
# 所以这个 CPD 表示：
# P(Grade | Intelligence, Difficulty)
#
# 也就是说：
# 在不同智力水平和课程难度下，
# 成绩为 A、B、C 的概率分别是多少。
#
# variable_card=3：
# Grade 有 3 个状态：
# Grade=0：A
# Grade=1：B
# Grade=2：C
#
# evidence=["Intelligence", "Difficulty"]：
# 父节点顺序是：
# 第 1 个父节点：Intelligence
# 第 2 个父节点：Difficulty
#
# evidence_card=[2, 2]：
# Intelligence 有 2 个状态。
# Difficulty 有 2 个状态。
#
# 因此一共有：
# 2 × 2 = 4 列
# ------------------------------------------------------------
cpd_grade = TabularCPD(
    variable="Grade",
    variable_card=3,

    # 列顺序由证据顺序决定：
    # evidence=["Intelligence", "Difficulty"]
    #
    # 第 1 列：(Intelligence=0, Difficulty=0)
    # 也就是：智力不高，课程容易。
    #
    # 第 2 列：(Intelligence=0, Difficulty=1)
    # 也就是：智力不高，课程难。
    #
    # 第 3 列：(Intelligence=1, Difficulty=0)
    # 也就是：智力高，课程容易。
    #
    # 第 4 列：(Intelligence=1, Difficulty=1)
    # 也就是：智力高，课程难。
    values=[
        # 第 1 行对应 Grade=0，也就是成绩为 A。
        # 四个数字依次表示：
        # 智力不高、课程容易：A 的概率是 0.30。
        # 智力不高、课程难：A 的概率是 0.05。
        # 智力高、课程容易：A 的概率是 0.90。
        # 智力高、课程难：A 的概率是 0.50。
        [0.30, 0.05, 0.90, 0.50],

        # 第 2 行对应 Grade=1，也就是成绩为 B。
        # 四个数字依次表示：
        # 智力不高、课程容易：B 的概率是 0.40。
        # 智力不高、课程难：B 的概率是 0.25。
        # 智力高、课程容易：B 的概率是 0.08。
        # 智力高、课程难：B 的概率是 0.30。
        [0.40, 0.25, 0.08, 0.30],

        # 第 3 行对应 Grade=2，也就是成绩为 C。
        # 四个数字依次表示：
        # 智力不高、课程容易：C 的概率是 0.30。
        # 智力不高、课程难：C 的概率是 0.70。
        # 智力高、课程容易：C 的概率是 0.02。
        # 智力高、课程难：C 的概率是 0.20。
        [0.30, 0.70, 0.02, 0.20],
    ],

    # Grade 的两个父节点。
    evidence=["Intelligence", "Difficulty"],

    # 两个父节点各有 2 个状态。
    # 顺序必须与 evidence 一致。
    evidence_card=[2, 2],
)


# ------------------------------------------------------------
# 建立 Letter 节点的 CPD。
#
# Letter 有两个父节点：
# 1. Grade
# 2. SAT
#
# 所以它表示：
# P(Letter | Grade, SAT)
#
# 含义是：
# 在已知课程成绩和 SAT 成绩的情况下，
# 推荐信是普通推荐信还是强推荐信。
#
# variable_card=2：
# Letter=0：普通推荐信
# Letter=1：强推荐信
#
# Grade 有 3 个状态。
# SAT 有 2 个状态。
#
# 因此一共有：
# 3 × 2 = 6 列
# ------------------------------------------------------------
cpd_letter = TabularCPD(
    variable="Letter",
    variable_card=2,

    # 列顺序由 evidence=["Grade", "SAT"] 决定：
    #
    # 第 1 列：Grade=0(A)，SAT=0(低分)
    # 第 2 列：Grade=0(A)，SAT=1(高分)
    # 第 3 列：Grade=1(B)，SAT=0(低分)
    # 第 4 列：Grade=1(B)，SAT=1(高分)
    # 第 5 列：Grade=2(C)，SAT=0(低分)
    # 第 6 列：Grade=2(C)，SAT=1(高分)
    values=[
        # 第 1 行对应 Letter=0，也就是普通推荐信。
        # 六个数字依次对应上面六列条件。
        # 成绩 A、SAT 低：普通推荐信概率是 0.10。
        # 成绩 A、SAT 高：普通推荐信概率是 0.40。
        # 成绩 B、SAT 低：普通推荐信概率是 0.60。
        # 成绩 B、SAT 高：普通推荐信概率是 0.85。
        # 成绩 C、SAT 低：普通推荐信概率是 0.95。
        # 成绩 C、SAT 高：普通推荐信概率是 0.99。
        [0.10, 0.40, 0.60, 0.85, 0.95, 0.99],

        # 第 2 行对应 Letter=1，也就是强推荐信。
        # 六个数字依次对应上面六列条件。
        # 成绩 A、SAT 低：强推荐信概率是 0.90。
        # 成绩 A、SAT 高：强推荐信概率是 0.60。
        # 成绩 B、SAT 低：强推荐信概率是 0.40。
        # 成绩 B、SAT 高：强推荐信概率是 0.15。
        # 成绩 C、SAT 低：强推荐信概率是 0.05。
        # 成绩 C、SAT 高：强推荐信概率是 0.01。
        [0.90, 0.60, 0.40, 0.15, 0.05, 0.01],
    ],

    # Letter 的两个父节点。
    evidence=["Grade", "SAT"],

    # Grade 有 3 个状态，SAT 有 2 个状态。
    # 顺序必须和 evidence 一一对应。
    evidence_card=[3, 2],
)


# ------------------------------------------------------------
# 把前面创建的 5 个 CPD 加入模型中。
#
# 加入之后，每个节点就有了自己的概率规则：
# Intelligence：先验概率。
# Difficulty  ：先验概率。
# SAT         ：P(SAT | Intelligence)。
# Grade       ：P(Grade | Intelligence, Difficulty)。
# Letter      ：P(Letter | Grade, SAT)。
#
# 如果某个节点没有 CPD，模型通常就是不完整的。
# ------------------------------------------------------------
model.add_cpds(cpd_i, cpd_d, cpd_sat, cpd_grade, cpd_letter)


# ------------------------------------------------------------
# 检查贝叶斯网络是否完整、合法。
#
# check_model() 通常会检查：
# 1. 每个节点是否都有 CPD。
# 2. CPD 的变量名是否和模型节点名一致。
# 3. CPD 中的父节点是否和图中的边一致。
# 4. 状态数量是否匹配。
# 5. 每列概率是否合法。
# 6. 概率是否满足归一化条件。
#
# 如果模型正确，通常返回 True。
# 如果有明显问题，可能返回 False 或直接报错。
#
# 注意：
# 这一行没有写 print，
# 所以即使模型检查通过，终端可能也不会显示 True。
# ------------------------------------------------------------
model.check_model()


# ------------------------------------------------------------
# 打印特定节点的 CPD。
#
# 这里指定取出 Letter 节点的 CPD。
# get_cpds("Letter") 返回 Letter 对应的 CPD 对象。
# print 会把这个对象格式化成概率表显示出来。
#
# 因为没有设置 state_names，
# 输出中可能会显示 Letter(0)、Letter(1)、Grade(0) 等数字状态。
# ------------------------------------------------------------
# 打印特定的概率表，比如 Letter
print("Letter的条件概率表：")
print(model.get_cpds('Letter'))


# ------------------------------------------------------------
# 打印一条分隔线。
#
# "-" * 30 表示把短横线重复 30 次。
# 它只是为了让终端输出更清楚，没有计算意义。
# ------------------------------------------------------------
print("-" * 30)


# ------------------------------------------------------------
# 打印所有节点的条件概率表。
#
# model.get_cpds()：
# 不传节点名，返回模型中所有节点的 CPD 列表。
#
# for cpd in ...：
# 依次取出每一个 CPD，赋值给变量 cpd。
# 每次循环打印一个节点的概率表。
# ------------------------------------------------------------
print("所有节点的条件概率表：")
for cpd in model.get_cpds():

    # 打印当前节点的 CPD。
    # pgmpy 会把它格式化成像表格一样的内容。
    print(cpd)

    # 每个节点的 CPD 后面打印一条分隔线。
    print("-" * 20)