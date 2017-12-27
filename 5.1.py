# -------------------------------------------------------------- #
# from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# -------------------------------------------------------------- #
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# 5.1.1 scikit-learn의 교차 검증
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# 5.1.3 계층별 k-겹 교차 검증과 그외 전략들
from sklearn.datasets import load_iris
# LOOCV
from sklearn.model_selection import LeaveOneOut
# 임의 분할 교차 검증
from sklearn.model_selection import ShuffleSplit
# 그룹별 교차 검증
from sklearn.model_selection import GroupKFold
# -------------------------------------------------------------- #

# 인위적인 데이터셋을 만듭니다.
# X, y = make_blobs(random_state=0)
# 데이터와 타깃 레이블을 훈련 세트와 테스트 세트로 나눕니다.
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 모델 객체를 만들고 훈련 세트로 학습시킵니다.
# logreg = LogisticRegression().fit(X_train, y_train)
# 모델을 테스트 세트로 평가합니다.
# print("테스트 세트 점수 : {:.2f}".format(logreg.score(X_test, y_test)))

# 5.1 교차 검증
# mglearn.plots.plot_cross_validation()
# plt.show()

# 5.1.1 scikit-learn의 교차 검증
iris = load_iris()
logreg = LogisticRegression()
# 폴드 5개
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("교차 검증 점수      : {}".format(scores))
print("교차 검증 평균 점수 : {:.2f}".format(scores.mean()))

# 5.1.3 계층별 k-겹 교차 검증과 그외 전략들
iris = load_iris()
print("Iris lable : {}".format(iris.target))

# 계층별 k-겹 교차 검증
# mglearn.plots.plot_stratified_cross_validation()
# plt.show()

# 교차 검증 상세 옵션
from sklearn.model_selection import KFold
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print("교체 검증 점수 : {}".format(cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

# LOOCV
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("교차 검증 분할 횟수 : ", len(scores))
print("평균 정확도 : {:.2f} ".format(scores.mean()))

# 임의 분할 교차 검증
# mglearn.plots.plot_shuffle_split()
# plt.show()
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("교차 검증 점수 :\n{}".format(scores))

# 그룹별 교차 검증
# 인위적 데이터셋 생성
X, y = make_blobs(n_samples=12, random_state=0)
# 처음 세 개의 샘플은 같은 그룹에 속하고 다음은 네 개의 샘플이 같습니다.
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print("교차 검증 점수 : \n{}".format(scores))
mglearn.plots.plot_group_kfold()
plt.show()
