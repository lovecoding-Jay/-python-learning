import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. 使用训练集的中位数进行离散化
thresholds = np.median(X_train, axis=0)

def to_low_high(X):
    return (X >= thresholds).astype(int)

Xtr = to_low_high(X_train)
Xte = to_low_high(X_test)

# 4. 计算先验概率
classes = [0, 1, 2]

counts = {
    c: int((y_train == c).sum())
    for c in classes
}

priors = {
    c: counts[c] / len(y_train)
    for c in classes
}

# 5. 计算条件概率
likelihood = {}

for c in classes:
    Xc = Xtr[y_train == c]

    likelihood[c] = [
        float((Xc[:, f] == 1).sum()) / len(Xc)
        for f in range(X.shape[1])
    ]

# 6. 预测单个样本
def predict_one(x):
    best_c = None
    best_score = -float("inf")

    for c in classes:
        score = np.log(priors[c])

        for f in range(len(x)):
            if x[f] == 1:
                p = likelihood[c][f]
            else:
                p = 1.0 - likelihood[c][f]

            score += np.log(p)

        if score > best_score:
            best_score = score
            best_c = c

    return best_c

# 7. 预测并评估
preds = [predict_one(x) for x in Xte]

print("手写朴素贝叶斯准确率:",
      round(accuracy_score(y_test, preds), 4))