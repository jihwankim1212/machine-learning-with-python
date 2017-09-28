# -------------------------------------------------------------- #
# from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# -------------------------------------------------------------- #
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# 오차 행렬
from sklearn.metrics import confusion_matrix
# f-점수
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
# 불확실성 고려
from mglearn.datasets import make_blobs
# 정밀도-재현율 곡선과 ROC곡선
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
# 평균 정밀도 p354
from sklearn.metrics import average_precision_score
# ROC 와 AUC p355
from sklearn.metrics import roc_curve
# AUC area under the curve
from sklearn.metrics import roc_auc_score
# 5.3.3 다중 분류의 평가 지표
from sklearn.metrics import accuracy_score
from sklearn.metrics.scorer import SCORERS
# -------------------------------------------------------------- #

digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
# print("예측된 유니크 레이블 : {}".format(np.unique(pred_most_frequent)))
# print("테스트 점수 : {:.2f}".format(dummy_majority.score(X_test, y_test)))

# tree 테스트 점수
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
# print("테스트 점수 : {:.2f}".format(tree.score(X_test, y_test)))

# Dummy와 LogisticRegression 비교
dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
# print("dummy 점수 : {:.2f}".format(dummy.score(X_test, y_test)))
logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
# print("logreg 점수 : {:.2f}".format(logreg.score(X_test, y_test)))

# 오차 행렬
confusion = confusion_matrix(y_test, pred_logreg)
# print("오차 행렬 {}".format(confusion))

# print("반도 기반 더미 모델 : ")
# print(confusion_matrix(y_test, pred_most_frequent))
# print("\n무작위 더미 모델 : ")
# print(confusion_matrix(y_test, pred_dummy))
# print("\n결정 트리 :")
# print(confusion_matrix(y_test, pred_tree))
# print("\n로지스틱 회귀")
# print(confusion_matrix(y_test, pred_logreg))

# f-점수
# print("빈도 기반더미 모델의 f1 score : {:.2f}".format(f1_score(y_test, pred_most_frequent)))
# print("부작위 더미 모델의 f1 score : {:.2f}".format(f1_score(y_test, pred_dummy)))
# print("트리 모델의 f1 score : {:.2f}".format(f1_score(y_test, pred_tree)))
# print("로지스틱 회귀 모델의 f1 score : {:.2f}".format(f1_score(y_test, pred_logreg)))
# print(classification_report(y_test, pred_most_frequent, target_names=["9 아님", "9 맞음"]))
# print(classification_report(y_test, pred_dummy, target_names=["9 아님", "9 맞음"]))
# print(classification_report(y_test, pred_tree, target_names=["9 아님", "9 맞음"]))
# print(classification_report(y_test, pred_logreg, target_names=["9 아님", "9 맞음"]))

# 불확실성 고려
X, y = make_blobs(n_samples=(400, 50), centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
# mglearn.plots.plot_decision_threshold()
# plt.show()
# print(classification_report(y_test, svc.predict(X_test)))
# y_pred_lower_threshold = svc.decision_function(X_test) > -.8
# print(classification_report(y_test, y_pred_lower_threshold))

# 정밀도-재현율 곡선과 ROC곡선
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))
# 부드러운 곡선을 위해 데이터 포인트 수를 늘립니다.
X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))
# 0에 가까운 임계값을 찾습니다.
close_zero = np.argmin(np.abs(thresholds))
# plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10, label="threshold 0", fillstyle="none", c='k', mew=2)
# plt.plot(precision, recall, label="precision-reall curve")
# plt.xlabel("precision")
# plt.ylabel("recall")
# plt.legend()
# plt.show()

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)
# RandomForestClassifier는 decision_function 대신 predict_proba를 제공합니다.
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])
# plt.plot(precision, recall, label="svc")
# plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10, label="svc: threshold 0", fillstyle="none", c='k', mew=2)
# plt.plot(precision_rf, recall_rf, label="rf")
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
# plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', c='k', markersize=10, label="rf: threshold 0.5", fillstyle="none", mew=2)
# plt.xlabel("precision")
# plt.ylabel("recall")
# plt.legend()
# plt.show()

# print("랜덤 포레스트의 f1_score : {:.3f}".format(f1_score(y_test, rf.predict(X_test))))
# print("svc의 f1_score : {:.3f}".format(f1_score(y_test, svc.predict(X_test))))

# 평균 정밀도 p354
ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
# print("랜덤 포레스트의 평균 정밀도 : {:.3f}".format(ap_rf))
# print("svc의 평균 정밀도 : {:.3f}".format(ap_svc))

# ROC 와 AUC p355
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
# plt.plot(fpr, tpr, label="ROC Curve")
# plt.xlabel("FPR")
# plt.ylabel("TPR")
# 0 근처의 임계값을 찾습니다.
close_zero = np.argmin(np.abs(thresholds))
# plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, label="threshold 0", fillstyle="none", c='k', mew=2)
# plt.legend()
# plt.show()

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
# plt.plot(fpr, tpr, label="SVC's ROC Curve")
# plt.plot(fpr_rf, tpr_rf, label="RF's ROC Curve")

# plt.xlabel("FPR")
# plt.ylabel("TPR")
# plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, label="SVC threshold 0", fillstyle="none", c='k', mew=2)
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
# plt.plot(fpr_rf[close_default_rf], tpr[close_default_rf], '^', markersize=10, label="RF threshold 0.5", fillstyle="none", c='k', mew=2)
# plt.legend()
# plt.show()

# AUC area under the curve
# rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
# svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
# print("랜덤 포레스트의 AUC {:.3f}".format(rf_auc))
# print("SVC의 AUC {:.3f}".format(svc_auc))

y = digits.target == 9
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)
# plt.figure()
# for gamma in [1, 0.1, 0.01]:
#     svc = SVC(gamma=gamma).fit(X_train, y_train)
#     accuracy = svc.score(X_test, y_test)
#     auc = roc_auc_score(y_test, svc.decision_function(X_test))
#     fpr, tpr, _ = roc_curve(y_test, svc.decision_function(X_test))
#     print("gamma = {:.2f} 정확도 = {:.2f} AUC = {}".format(gamma, accuracy, auc))
#     plt.plot(fpr, tpr, label="gamma={:.3f}".format(gamma))
# plt.xlabel("FPR")
# plt.ylabel("TPR")
# plt.xlim(-0.01, 1)
# plt.ylim(0, 1.02)
# plt.legend()
# plt.show()

# 5.3.3 다중 분류의 평가 지표
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0)
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
# print("정확도 : {:.3f}".format(accuracy_score(y_test, pred)))
# print("오차 행렬 : \n{}".format(confusion_matrix(y_test, pred)))
scores_image = mglearn.tools.heatmap(confusion_matrix(y_test, pred), xlabel='predict label', ylabel='real label', xticklabels=digits.target_names, yticklabels=digits.target_names, cmap=plt.cm.gray_r, fmt="%d")
# plt.title("confusion matrix")
# plt.gca().invert_yaxis()
# plt.show()
# print(classification_report(y_test, pred))
# print("micro 평균 f1 점수 : {:.3f}".format(f1_score(y_test, pred, average="micro")))
# print("macro 평균 f1 점수 : {:.3f}".format(f1_score(y_test, pred, average="macro")))

# 5.3.5 모델 선택에서 평가 지표 사용하기
# 분류의 기본 평가 지표는 정확도입니다.
print("기본 평가 지표 : {}".format(cross_val_score(SVC(), digits.data, digits.target == 9)))
# scoring="accuracy"의 결과는 같습니다.
explicit_accuracy = cross_val_score(SVC(), digits.data, digits.target == 9, scoring="accuracy")
print("정확도 지표 : {}".format(explicit_accuracy))
roc_auc = cross_val_score(SVC(), digits.data, digits.target == 9, scoring="roc_auc")
print("AUC 지표 : {}".format(roc_auc))

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target == 9, random_state=0)
# 일부러 적절하지 않은 그리드를 만듭니다.
param_grid = {'gamma' : [0.0001, 0.01, 0.1, 1, 10]}
# 기본값인 정확도 평가 지표를 사용합니다.
grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)
print("정확도 지표를 사용한 그리드 서치")
print("최적의 매개변수 : ",grid.best_params_)
print("최상의 교차 검증 점수 (정확도)) : {:.3f}".format(grid.best_score_))
print("테스트 세트 AUC : {:.3f}".format(roc_auc_score(y_test, grid.decision_function(X_test))))
print("테스트 세트 정확도 : {:.3f}\n".format(grid.score(X_test, y_test)))

# AUC 지표 사용
grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="roc_auc")
grid.fit(X_train, y_train)
print("AUC 지표를 사용한 그리드 서치")
print("최적의 매개변수 :", grid.best_params_)
print("최상의 교차 검증 점수 (AUC) : {}".format(grid.best_score_))
print("테스트 세트 AUC : {}".format(roc_auc_score(y_test, grid.decision_function(X_test))))
print("테스트 세트 정확도 : {}\n".format(grid.score(X_test, y_test)))

print("가능한 평가 방식 :\n {}".format(sorted(SCORERS.keys())))
