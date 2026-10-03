try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([
    ("Intelligence", "SAT"),
    ("Intelligence", "Grade"),
    ("Difficulty", "Grade"),
    ("Grade", "Letter"),
    ("SAT", "Letter"),
])

cpd_i = TabularCPD(
    variable="Intelligence",
    variable_card=2,
    values=[[0.7], [0.3]],
)

cpd_d = TabularCPD(
    variable="Difficulty",
    variable_card=2,
    values=[[0.6], [0.4]],
)

cpd_sat = TabularCPD(
    variable="SAT",
    variable_card=2,
    values=[
        [0.95, 0.20],
        [0.05, 0.80],
    ],
    evidence=["Intelligence"],
    evidence_card=[2],
)

cpd_grade = TabularCPD(
    variable="Grade",
    variable_card=3,
    values=[
        [0.30, 0.05, 0.90, 0.50],
        [0.40, 0.25, 0.08, 0.30],
        [0.30, 0.70, 0.02, 0.20],
    ],
    evidence=["Intelligence", "Difficulty"],
    evidence_card=[2, 2],
)

cpd_letter = TabularCPD(
    variable="Letter",
    variable_card=2,
    values=[
        [0.10, 0.40, 0.60, 0.85, 0.95, 0.99],
        [0.90, 0.60, 0.40, 0.15, 0.05, 0.01],
    ],
    evidence=["Grade", "SAT"],
    evidence_card=[3, 2],
)

model.add_cpds(cpd_i, cpd_d, cpd_sat, cpd_grade, cpd_letter)
model.check_model()
# 打印特定的概率表，比如 Letter
print("Letter的条件概率表：")
print(model.get_cpds('Letter'))

print("-" * 30)

# 打印所有的 CPD
print("所有节点的条件概率表：")
for cpd in model.get_cpds():
    print(cpd)
    print("-" * 20)