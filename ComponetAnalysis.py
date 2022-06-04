import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA

X = np.array([[10, 12, 14], [12, 18, 17], [11, 17, 19]])
print(X.shape)

degree = 2
n_components = 3
poly = PolynomialFeatures(degree)
x_poly = poly.fit_transform(X)
print('Poly shape: ', x_poly.shape)

pca = PCA(n_components=n_components)
x_pca_poly = pca.fit_transform(x_poly)
print(x_pca_poly)

kern_obj = KernelPCA(n_components=n_components, degree=degree, kernel='poly')
kern_obj.fit(X)
x_kern = kern_obj.transform(X)
print('kernel pca shape: ', x_kern)
