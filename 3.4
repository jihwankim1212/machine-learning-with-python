# -------------------------------------------------------------- #
# from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# -------------------------------------------------------------- #
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# 고유얼굴(eigenface) 특성 추출 p187
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
# -------------------------------------------------------------- #

# 3.4 차원 축소, 특성 추출, 매니폴드 학습
# mglearn.plots.plot_pca_illustration()
# plt.show()

# 주성분 분석 (PCA)
cancer = load_breast_cancer()
# fig, axes = plt.subplots(15, 2, figsize=(10,20))
# malignant = cancer.data[cancer.target == 0]
# benign    = cancer.data[cancer.target == 1]
# ax = axes.ravel()
# for i in range(30):
#     _, bins = np.histogram(cancer.data[:, i], bins=50)
#     ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
#     ax[i].hist(benign[:, i],    bins=bins, color=mglearn.cm3(2), alpha=.5)
#     ax[i].set_title(cancer.feature_names[i])
#     ax[i].set_yticks(())
# ax[0].set_xlabel("size")
# ax[1].set_ylabel("att 0")
# ax[0].legend(["bad","good"], loc="best")
# fig.tight_layout()
# plt.show()

# scaler = StandardScaler()
# scaler.fit(cancer.data)
# X_scaled = scaler.transform(cancer.data)

# 데이터의 처음 두 개의 주성분만 유지합니다.
# pca = PCA(n_components=2)
# 유방암 데이터로 PCA 모델을 만듭니다.
# pca.fit(X_scaled)
# 처음 두 개의 주성분을 사용해 데이터를 변환합니다.
# X_pca = pca.transform(X_scaled)
# print("  원본 데이터 형태 : {}".format(str(X_scaled.shape)))
# print("축소된 데이터 형태 : {}".format(str(X_pca.shape   )))
# 클래스를 색깔로 구분하여 처음 두 개의 주성분을 그래프로 나타냅니다.
# plt.figure(figsize=(8,8))
# mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
# plt.legend(["bad", "good"], loc="best")
# plt.gca().set_aspect("equal")
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.show()
# print("PCA 주성분 형태 : {}".format(pca.components_.shape))
# print("PCA 주성분      : {}".format(pca.components_))
# plt.matshow(pca.components_, cmap='viridis')
# plt.yticks([0, 1], ["1", "2"])
# plt.colorbar()
# plt.xticks(range(len(cancer.feature_names)),
#            cancer.feature_names, rotation=60, ha='left')
# plt.xlabel("att")
# plt.ylabel("pca")
# plt.show()

# 고유얼굴(eigenface) 특성 추출 p187
# people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
# image_shape = people.images[0].shape
# fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks' : (), 'yticks' : ()})
# for target, image, ax in zip(people.target, people.images, axes.ravel()):
#     ax.imshow(image)
#     ax.set_title(people.target_names[target])
# plt.show()
# print("people.images.shape : {}".format(people.images.shape))   # image 3023, 87 X 65 픽셀
# print("class 개수 : {}".format(len(people.target_names)))
# 각 타깃이 나타난 횟수 계산
# counts = np.bincount(people.target)
# 타깃별 이름과 횟수 출력
# for i, (count, name) in enumerate(zip(counts, people.target_names)) :
#     print("{0:25} {1:3}".format(name, count), end='   ')
#     if (i + 1) % 2 == 0:
#         print()
# mask = np.zeros(people.target.shape, dtype=np.bool)
# for target in np.unique(people.target):
#     mask[np.where(people.target == target)[0][:50]] = 1
# X_people = people.data[mask]
# y_people = people.target[mask]
# 0~255 사이의 흑백 이미지의 픽셀 값을 0~1 스케일로 조정합니다.
# MinMaxScaler를 적용하는 것과 같습니다.
# X_people = X_people / 255.
# KNeighborsClassifier 적용
# 데이터를 훈련 세트와 테스트 세트로 나눕니다.
# X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
# 이웃 개수를 한 개로 하여 KNeighborsClassifier 모델을 만듭니다.
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)
# print("1 - 최근접 이웃의 테스트 세트 점수 : {:.2f}".format((knn.score(X_test, y_test))))
# mglearn.plots.plot_pca_whitening()
# plt.show()
# pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
# X_train_pca = pca.transform(X_train)
# X_test_pca  = pca.transform(X_test)
# print("X_train_pca.shape : {}".format(X_train_pca.shape))
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train_pca, y_train)
# print("테스트 세트 정확도 : {:.2f}".format(knn.score(X_test_pca, y_test)))
# print("pca.components_.shape : {}".format(pca.components_.shape))
# fig, axes = plt.subplots(3, 5, figsize = (15, 12), subplot_kw={'xticks' : (), 'yticks' : ()})
# for i, (componet, ax) in enumerate(zip(pca.components_, axes.ravel())):
#     ax.imshow(componet.reshape(image_shape), cmap='viridis')
#     ax.set_title("main component {}".format(i+1))
# plt.show()
# mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)
# plt.show()
# mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
# plt.xlabel("first component")
# plt.ylabel("second component")
# plt.show()

# 비음수 행렬 분해 (NMF) p197
# mglearn.plots.plot_nmf_illustration()
# plt.show()
# mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
# plt.show()

# nmf = NMF(n_components=15, random_state=0)
# nmf.fit(X_train)
# X_train_nmf = nmf.transform(X_train)
# X_test_nmf  = nmf.transform(X_test)
# fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks' : (), 'yticks' : ()})
# for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
#     ax.imshow(component.reshape(image_shape))
#     ax.set_title("component {}".format(i))
# plt.show()

# compn = 3
# 4번째 성분으로 정렬하여 처음 10개 이미지를 출력합니다.
# inds = np.argsort(X_train_nmf[:, compn])[::-1]
# fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks' : (), 'yticks' : ()})
# for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
#     ax.imshow(X_train[ind].reshape(image_shape))
# compn = 7
# 8번째 성분으로 정렬하여 처음 10개 이미지를 출력합니다.
# inds = np.argsort(X_train_nmf[:, compn])[::-1]
# fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks' : (), 'yticks' : ()})
# for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
#     ax.imshow(X_train[ind].reshape(image_shape))
# plt.show()

# S = mglearn.datasets.make_signals()
# plt.figure(figsize=(6, 1))
# plt.plot(S, '-')
# plt.xlabel("time")
# plt.ylabel("signal")
# plt.show()

# 원본 데이터를 사용해 100개의 측정 데이터를 만듭니다.
# A = np.random.RandomState(0).uniform(size=(100, 3))
# X = np.dot(S, A.T)
# print("측정 데이터 형태 : {}".format(X.shape))

# NMF 처리
# nmf = NMF(n_components=3, random_state=42)
# S_ = nmf.fit_transform(X)
# print("복원한 신호 데이터 형태 : {}".format(S_.shape))

# PCA 처리
# pca = PCA(n_components=3)
# H = pca.fit_transform(X)
#
# models = [X, S, S_, H]
# names  = ['측정 신호 (처음 3개)', '원본 신호', 'NMF로 복원한 신호', 'PCA로 복원한 신호']
# fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5}, subplot_kw={'xticks' : (), 'yticks' : ()})
# for model, name, ax in zip(models, names, axes):
#     ax.set_title(name)
#     ax.plot(model[:, :3], '-')
# plt.show()

digits = load_digits()
# fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks' : (), 'yticks' : ()})
# for ax, img in zip(axes.ravel(), digits.images):
#     ax.imshow(img)
# plt.show()
print("digits {}".format(digits.data))

# PCA 모델을 생성합니다.
pca = PCA(n_components=2)
pca.fit(len(digits.data))
# 처음 두 개의 주성분으로 숫자 데이터를 변환합니다.
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120", "#535D83"]
# plt.figure(figsize=(10, 10))
# plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
# plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
# for i in range(len(digits.data)):
    # 숫자 텍스트를 이용해 산점도를 그립니다.
#     plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), color = colors[digits.target[i]], fontdict={'weight' : 'bold', 'size' : 9})
# plt.xlabel("first component")
# plt.ylabel("second component")
# plt.show()

tsne = TSNE(random_state=42)
# TSNE에는 transform 메서드가 없으므로 대신 fit_transform을 사용합니다.
digits_tsne = tsne.fit_transform(digits.data)
plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1 )
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1 )
for i in range(len(digits.data)):
    # 숫자 텍스트를 이용해 산점도를 그립니다.
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), color = colors[digits.target[i]], fontdict={'weight' : 'bold', 'size' : 9})
plt.xlabel("t-SNE att 0")
plt.ylabel("t-SNE att 1")
plt.show()
