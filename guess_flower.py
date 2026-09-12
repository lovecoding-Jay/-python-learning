import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#  加载数据：X 为 4 维特征，y 为类别标签 0/1/2
iris = load_iris()
X, y = iris.data, iris.target

#  划分训练集/测试集（禁止用测试集训练）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

#  离散化：用训练集各特征的中位数作为阈值
thresholds = np.median(X_train, axis=0)

def to_low_high(X):
    return (X >= thresholds).astype(int)   # 1=高，0=低

Xtr, Xte = to_low_high(X_train), to_low_high(X_test)

#  类别先验 P(c)：各类别在训练集中的占比
classes = [0, 1, 2]
counts = {c: int((y_train == c).sum()) for c in classes}
priors = {c: counts[c] / len(y_train) for c in classes}

#  条件概率 P(x_i=高 | c)：数频率
likelihood = {}
for c in classes:
    Xc = Xtr[y_train == c]
    likelihood[c] = [float((Xc[:, f] == 1).sum()) / len(Xc)
                     for f in range(X.shape[1])]

#  预测：比较 log P(c) + Σ log P(x_i | c)
def predict_one(x):
    best_c, best_score = None, -float("inf")
    for c in classes:
        score = np.log(priors[c])
        for f in range(len(x)):
            p = likelihood[c][f] if x[f] == 1 else 1.0 - likelihood[c][f]
            score += np.log(p)
        if score > best_score:
            best_score, best_c = score, c
    return best_c

#  在测试集上评估
preds = [predict_one(x) for x in Xte]
print("手写朴素贝叶斯准确率:", round(accuracy_score(y_test, preds), 4))