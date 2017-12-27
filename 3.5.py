# -------------------------------------------------------------- #
# from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# -------------------------------------------------------------- #
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# two_moons 데이터를 생성합니다(이번에는 노이즈를 조금만 넣습니다) p217
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
# 병합 군집 p226
from sklearn.cluster import AgglomerativeClustering
# 덴드로그램 p228
# SciPy에서 ward 군집 함수와 덴드로그램 함수를 임포트합니다.
from scipy.cluster.hierarchy import  dendrogram, ward
# DBSCAN p231
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# 군집 알고리즘의 비교와 평가 p234
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
# 타깃값 없이 군집 평가하기 p236
from sklearn.metrics.cluster import silhouette_score
# 얼굴 데이터셋으로 군집 알고리즘 비교 p239
# LFW 데이터에서 고유얼굴을 찾은 다음 데이터를 변환합니다.
from sklearn.decomposition import PCA
# -------------------------------------------------------------- #


# 3.5.1 k-평균 군집
# mglearn.plots.plot_kmeans_algorithm()
# plt.show()
# mglearn.plots.plot_kmeans_boundaries()
# plt.show()

# 인위적으로 2차원 데이터를 생성합니다.
# X, y = make_blobs(random_state=1)
# 군집 모델을 만듭니다.
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# print("클러스터 레이블 :\n {}".format(kmeans.labels_))
# print(kmeans.predict(X))
# mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
# mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)
# plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# 두 개의 클러스터 중심을 사용합니다.
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(X)
# assignments = kmeans.labels_
# mglearn.discrete_scatter(X[:, 0],X[:, 1], assignments, ax=axes[0])

# 다섯 개의 클러스터 중심을 사용합니다.
# kmeans = KMeans(n_clusters=5)
# kmeans.fit(X)
# assignments = kmeans.labels_
# mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])
# plt.legend()
# plt.show()

# X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5], random_state=170)
# y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)
# mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
# plt.legend()
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.show()

# 무작위로 클러스터 데이터를 생성합니다. p216
# X, y = make_blobs(random_state=170, n_samples=600)
# rng = np.random.RandomState(74)
# 데이터가 길게 늘어지도록 변경합니다.
# transformation = rng.normal(size=(2, 2))
# X = np.dot(X, transformation)
# 세 개의 클러스터로 데이터에 KMean 알고리즘을 적용합니다.
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# y_pred = kmeans.predict(X)
# 클러스터 할당과 클러스터 중심을 나타냅니다.
# mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
# mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.legend()
# plt.show()

# two_moons 데이터를 생성합니다(이번에는 노이즈를 조금만 넣습니다) p217
# X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# 두 개의 클러스터로 데이터에 KMeans 알고리즘을 적용합니다.
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(X)
# y_pred = kmeans.predict(X)
# 클러스터 할당과 클러스터 중심을 표시합니다.
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm2, s=60, edgecolors='k')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidths=2, edgecolors='k')
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# -------------------------------------------------------------- #
# people
# -------------------------------------------------------------- #
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
# fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks' : (), 'yticks' : ()})
# for target, image, ax in zip(people.target, people.images, axes.ravel()):
#     ax.imshow(image)
#     ax.set_title(people.target_names[target])
# plt.show()
# print("people.images.shape : {}".format(people.images.shape))   # image 3023, 87 X 65 픽셀
# print("class 개수 : {}".format(len(people.target_names)))
# 각 타깃이 나타난 횟수 계산
counts = np.bincount(people.target)
# 타깃별 이름과 횟수 출력
# for i, (count, name) in enumerate(zip(counts, people.target_names)) :
#     print("{0:25} {1:3}".format(name, count), end='   ')
#     if (i + 1) % 2 == 0:
#         print()
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
# 0~255 사이의 흑백 이미지의 픽셀 값을 0~1 스케일로 조정합니다.
# MinMaxScaler를 적용하는 것과 같습니다.
X_people = X_people / 255.
# KNeighborsClassifier 적용
# 데이터를 훈련 세트와 테스트 세트로 나눕니다.
# -------------------------------------------------------------- #
# people
# -------------------------------------------------------------- #

# X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
# nmf = NMF(n_components=100, random_state=0)         # NMF
# nmf.fit(X_train)
# pca = PCA(n_components=100, random_state=0)         # PCA
# pca.fit(X_train)
# kmeans = KMeans(n_clusters=100, random_state=0)     # KMean
# kmeans.fit(X_train)
# X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
# X_reconstructed_kmeas = kmeans.cluster_centers_[kmeans.predict(X_test)]
# X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)

# components_
# fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks' : (), 'yticks' : ()})
# fig.suptitle("component")
# for ax, comp_kmeans, comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
#     ax[0].imshow(comp_kmeans.reshape(image_shape))
#     ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
#     ax[2].imshow(comp_nmf.reshape(image_shape))
# axes[0, 0].set_ylabel("kmeans")
# axes[1, 0].set_ylabel("pca")
# axes[2, 0].set_ylabel("nmf")

# reconstruct
# fig, axes = plt.subplots(4, 5, subplot_kw={'xticks' : (), 'yticks' : ()}, figsize=(8, 8))
# fig.suptitle("reconstruct")
# for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(axes.T, X_test, X_reconstructed_kmeas, X_reconstructed_pca, X_reconstructed_nmf):
#     ax[0].imshow(orig.reshape(image_shape))
#     ax[1].imshow(rec_kmeans.reshape(image_shape))
#     ax[2].imshow(rec_pca.reshape(image_shape))
#     ax[3].imshow(rec_nmf.reshape(image_shape))
# axes[0, 0].set_ylabel("original")
# axes[1, 0].set_ylabel("kmeans")
# axes[2, 0].set_ylabel("pca")
# axes[3, 0].set_ylabel("nmf")
# plt.show()

# X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# kmeans = KMeans(n_clusters=10, random_state=0)
# kmeans.fit(X)
# y_pred = kmeans.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired', edgecolors='black')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60, marker='^', c=range(kmeans.n_clusters), linewidths=2, cmap='Paired', edgecolors='black')
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# print("클러스터 레이블 :\n {}".format(y_pred))
# plt.show()
# distance_features = kmeans.transform(X)
# print("클러스터 거리 데이터의 형태 : {}".format(distance_features.shape))
# print("클러스터 거리 : \n {}".format(distance_features))

# 병합 군집 p226
# X, y = make_blobs(random_state=1)
# agg = AgglomerativeClustering(n_clusters=2)
# assignment = agg.fit_predict(X)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
# plt.legend()
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.show()

# 계층적 군집 p227
# mglearn.plots.plot_agglomerative()
# plt.show()

# 덴드로그램 p228
# SciPy에서 ward 군집 함수와 덴드로그램 함수를 임포트합니다.
# from scipy.cluster.hierarchy import  dendrogram, ward
# X, y = make_blobs(random_state=0, n_samples=12)
# 데이터 배열 x에 ward 함수를 적용합니다.
# SciPy의 ward 함수는 병합 군집을 수행할 때 생성된 거리 정보가 담긴 배열을 반환합니다.
# linkage_array = ward(X)
# 클러스터 간의 거리 정보가 담긴 linkage_array를 사용해 덴드로그램을 그립니다.
# dendrogram(linkage_array)

# 두 개와 세 개의 클러스터를 구분하는 커트라인을 표시합니다.
# ax = plt.gca()
# bounds = ax.get_xbound()
# ax.plot(bounds, [7.25, 7.25], '--', c='k')
# ax.plot(bounds, [4, 4], '--', c='k')
#
# ax.text(bounds[1], 7.25, ' two cluster', va='center', fontdict={'size' : 10 })
# ax.text(bounds[1], 4, 'three cluster', va='center', fontdict={'size' : 10})
# plt.xlabel("sample number")
# plt.ylabel("cluster distance")
# plt.show()

# DBSCAN p231
# X, y = make_blobs(random_state=0, n_samples=12)
# dbscan = DBSCAN()
# clusters = dbscan.fit_predict(X)
# print("클러스터 레이블 : \n {}".format(clusters))
# mglearn.plots.plot_dbscan()
# plt.show()

# X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# 평균이 0, 분산이 1이 되도록 데이터의 스케일을 조정합니다. StandardScaler()
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
# dbscan = DBSCAN()
# clusters = dbscan.fit_predict(X_scaled)
# 클러스터 할당을 표시합니다.
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60, edgecolors='black')
# plt.xlabel("att 0")
# plt.ylabel("att 1")
# plt.show()

# 군집 알고리즘의 비교와 평가 p234
# X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# 평균이 0, 분산이 1이 되도록 데이터의 스케일을 조정합니다.
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)

# fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks' : (), 'yticks' : ()})
# 사용할 알고리즘 모델을 리스트로 만듭니다.
# algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]
# 비교를 위해 무작위로 클러스터를 할당합니다.
# random_state = np.random.RandomState(seed=0)
# random_clusters = random_state.randint(low=0, high=2, size=len(X))
# 무작위로 할당한 클러스터를 그립니다.
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
# axes[0].set_title("random assign- ARI : {:.2f}".format(adjusted_rand_score(y, random_clusters)))

# for ax, algorithm in zip(axes[1:], algorithms):
    # 클러스터 할당과 클러스터 중심을 그립니다.
#     clusters = algorithm.fit_predict(X_scaled)
#     ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
#     ax.set_title("{} - ARI : {:.2f}".format(algorithm.__class__.__name__,adjusted_rand_score(y, clusters)))
# plt.show()

# 군집 평가하기
# 포인터가 클러스터로 나뉜 두 가지 경우
# cluster1 = [0, 0, 1, 1, 0]
# cluster2 = [1, 1, 0, 0, 1]
# 모든 레이블이 달라졌으므로 정확도는 0입니다.
# print("정확도 : {:.2f}".format(accuracy_score(cluster1, cluster2)))
# 같은 포인트가 한 클러스터에 모였으므로 ARI는 1입니다.
# print("ARI : {:.2f}".format(adjusted_rand_score(cluster1, cluster2)))

# 타깃값 없이 군집 평가하기 p236
# X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# 평균이 0, 분산이 1이 되도록 데이터의 스케일을 조정합니다.
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
# fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
# 비교를 위해 무작위로 클러스터를 할당합니다.
# random_state = np.random.RandomState(seed=0)
# random_clusters = random_state.randint(low=0, high=2, size=len(X))
# 무작위로 할당한 클러스터를 그립니다.
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
# axes[0].set_title("random assign : {:.2f}".format(silhouette_score(X_scaled, random_clusters)))
# algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]
# for ax, algorithm in zip(axes[1:], algorithms):
#     clusters = algorithm.fit_predict(X_scaled)
#     # 클러스터 할당과 클러스트 중심을 그립니다.
#     ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
#     ax.set_title("{} : {:.2f}".format(algorithm.__class__.__name__, silhouette_score(X_scaled, clusters)))
# plt.show()

# 얼굴 데이터셋으로 군집 알고리즘 비교 p239
# LFW 데이터에서 고유얼굴을 찾은 다음 데이터를 변환합니다.
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)
# 기본 매개변수로 DBSCAN을 적용합니다.
dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
# print("고유한 레이블 : {}".format(np.unique(labels)))
# 잡음 포인트와 클러스터에 속한 포인트 수를 셉니다.
# bincount는 음수를 받을 수 없어서 labels에 1을 더했습니다.
# 반환값의 첫 번째 원소는 잡음 포인트의 수입니다.
# print("클러스터별 포인트 수 : {} ".format(np.bincount(labels + 1)))
# noise = X_people[labels==-1]  # 잡음 선택
# fig, axes = plt.subplots(4, 9, subplot_kw={'xticks' : (), 'yticks' : ()}, figsize=(12, 4))
# for image, ax in zip(noise, axes.ravel()):
#     ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
# plt.show()

# eps 변화에 따른 클러스터별 포인트
# for eps in [1, 3, 5, 7, 9, 11, 13]:
#     print("\neps = {}".format(eps))
#     dbscan = DBSCAN(eps=eps, min_samples=3)
#     labels = dbscan.fit_predict(X_pca)
#     print("클러스터 수   : {}".format(len(np.unique(labels))))
#     print("클러스터 크기 : {}".format(np.bincount(labels + 1)))

# dbscan = DBSCAN(min_samples=3, eps=7)
# labels = dbscan.fit_predict(X_pca)
# for cluster in range(max(labels) + 1):
#     mask = labels == cluster
#     n_images = np.sum(mask)
#     fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4), subplot_kw={'xticks': (), 'yticks': ()})
#     for image, label, ax in zip(X_people[mask], y_people[mask], axes):
#         ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
#         ax.set_title(people.target_names[label].split()[-1])
# plt.show()

# K-평균으로 얼굴 데이터셋 분석하기 p245
# K-평균으로 클러스터를 추출합니다.
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
# print("K-평균의 클러스터 크기 : {}".format(np.bincount(labels_km)))
# fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(12, 4))
# for center, ax in zip(km.cluster_centers_, axes.ravel()):   # 10개의 cluster center
#     ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)
# plt.show()
# mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)
# plt.show()

# 병합 군집으로 얼굴 데이터셋 분석하기 p248
# 병합 군집으로 클러스터를 추출합니다.
# agglomerative = AgglomerativeClustering(n_clusters=10)
# labels_agg = agglomerative.fit_predict(X_pca)
# print("병합 군집의 클러스터 크기 : {}".format(np.bincount(labels_agg)))

# 병합 군집과 k-mean군집 비교
# print("ARI : {:.2f}".format(adjusted_rand_score(labels_agg, labels_km)))

# linkage_array = ward(X_pca)
# 클러스터 사이의 거리가 담겨있는 linkage_array로 덴드로그램을 그립니다.
# plt.figure(figsize=(20, 5))
# dendrogram(linkage_array, p=7, truncate_mode='level', no_labels=True)
# plt.xlabel("sample number")
# plt.ylabel("cluster distance")
# ax = plt.gca()
# bounds = ax.get_xbound()
# ax.plot(bounds, [36, 36], '--', c='k')
# plt.show()

# n_clusters = 10
# for cluster in range(n_clusters):
#     mask = labels_agg == cluster
#     fig, axes = plt.subplots(1, 10, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(15, 8))
#     axes[0].set_ylabel(np.sum(mask))
#     for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes):
#         ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
#         ax.set_title(people.target_names[label].split()[-1], fontdict={'fontsize': 9})
# plt.show()

# 병합 군집으로 클러스터를 추출합니다.
agglomerative = AgglomerativeClustering(n_clusters=40)
labels_agg = agglomerative.fit_predict(X_pca)
print("병합 군집의 클러스터 크기 : {}".format(np.bincount(labels_agg)))
n_clusters = 40
for cluster in [10, 13, 19, 22, 36]: # 흥미로운 클러스터 몇 개를 골랐습니다.
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 15, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(15, 8))
    cluster_size = np.sum(mask)
    axes[0].set_ylabel("#{} : {}".format(cluster, cluster_size))
    for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1], fontdict={'fontsize': 9})
    for i in range(cluster_size, 15):
        axes[i].set_visible(False)
plt.show()
