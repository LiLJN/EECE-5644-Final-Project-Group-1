import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sys import float_info
from math import ceil, floor
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from scipy.stats import norm, multivariate_normal

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title


mydataset = pd.read_excel('PCA.xlsx')


print(mydataset)



N = len(mydataset.index)

X = mydataset.iloc[:, :-1].to_numpy()
qualities = mydataset.iloc[:, -1].to_numpy()
le = preprocessing.LabelEncoder()
le.fit(qualities)
labels = mydataset['Sibsp+Parch']

gmm = {}
gmm['priors'] = (mydataset.groupby(['Sibsp+Parch']).size() / N).to_numpy()

num_classes = len(gmm['priors'])


gmm['mu'] = mydataset.groupby(['Sibsp+Parch']).mean().to_numpy()
n = gmm['mu'].shape[1]

def regularized_cov(X, lambda_reg):
    n = X.shape[0]
    sigma = np.cov(X)
    sigma += lambda_reg * np.eye(n)

gmm['Sigma'] = np.array([regularized_cov(X[labels == l].T, (1/n)) for l in range(num_classes)])

N_per_l = np.array([sum(labels == l) for l in range(num_classes)])
print(N_per_l)



Lambda = np.ones((num_classes, num_classes)) - np.eye(num_classes)


def perform_erm_classification(X, Lambda, gmm_params, C):

    class_cond_likelihoods = np.array(
        [multivariate_normal.pdf(X, gmm_params['mu'][i], gmm_params['Sigma'][i]) for i in range(C)])


    class_priors = np.diag(gmm_params['priors'])

    class_posteriors = class_priors.dot(class_cond_likelihoods)


    risk_mat = Lambda.dot(class_posteriors)


    return np.argmin(risk_mat, axis=0)

decisions = perform_erm_classification(X, Lambda, gmm, num_classes)

# Simply using sklearn confusion matrix
print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, labels)
fig, ax = plt.subplots(figsize=(10, 10))
conf_display = ConfusionMatrixDisplay.from_predictions(decisions, labels, ax=ax,display_labels=['0', '1', '2', '3', '4', '5', '6'], colorbar=False)
plt.ylabel('Predicted Labels')
plt.xlabel('True Labels')

correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples))

prob_error = 1 - (correct_class_samples / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

fig = plt.figure(figsize=(10, 10))

ax_subset = fig.add_subplot(111, projection='3d')

unique_qualities = np.sort(mydataset['Sibsp+Parch'].unique())
for q in range(unique_qualities[0], unique_qualities[-1]):
    ax_subset.scatter(mydataset[mydataset['Sibsp+Parch'] == q]['Survived'],
                      mydataset[mydataset['Sibsp+Parch'] == q]['SEX'],
                      mydataset[mydataset['Sibsp+Parch'] == q]['Age'], label="Sibsp+Parch {}".format(q))

ax_subset.set_xlabel("Survived")
ax_subset.set_ylabel("SEX")
ax_subset.set_zlabel("Age")


plt.title("Sibsp+Parch of Features")
plt.legend()
plt.tight_layout()
plt.show()
