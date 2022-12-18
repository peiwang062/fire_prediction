import pandas as pd
import numpy as np
import random

data_A2D = pd.read_csv('A_D.csv')
data_E2H = pd.read_csv('E_H.csv')
data_X = pd.read_csv('X.csv')

data_A2D = data_A2D.to_numpy()[:,1:].astype(np.float32)
data_E2H = data_E2H.to_numpy()[:,1:].astype(np.float32)
data_feature = np.concatenate((data_A2D, data_E2H), axis=1)

# normalization
data_mean = np.mean(data_feature, axis=0)
data_std = np.std(data_feature, axis=0)
data_feature = (data_feature - data_mean) / data_std

data_X = data_X.to_numpy()[:,1:].astype(np.float32)
data_target = np.squeeze(data_X)

x, _, _, _ = np.linalg.lstsq(data_feature, data_X, rcond=None)

# here x is the coefficient we care about, we extract the one corresponding to H
print(x[-1])

# let suppose M = alpha * B
random.seed(1234)
alpha = random.random()  # just assign a random number

data_M = alpha*data_feature[:,1]
data_feature = np.concatenate((data_feature, data_M[:, np.newaxis]), axis=1)

# redo the fit
x, _, _, _ = np.linalg.lstsq(data_feature, data_X, rcond=None)



# --------- compute the pearson correlation coefficient ---------------
from scipy.stats import pearsonr
for i in range(np.shape(data_feature)[1]):
    corr, p_value = pearsonr(np.squeeze(data_feature[:,i]), np.squeeze(data_X))
    print(corr, p_value)



# We first define the prior and likelihood
P_equities = 0.7
P_options = 0.2
P_crypto = 0.1
P_fraudulent_given_equities = 0.01
P_fraudulent_given_options = 0.03
P_fraudulent_given_crypto = 0.07

# Q1
# This problem can be solved by bayes rule. The probability of options given fraudulent can be computed by the likelihood of fraudulent given options，the prior of the options and those for equities and crypto
P_options_given_fraudulent = (P_fraudulent_given_options * P_options) / (P_fraudulent_given_equities * P_equities + P_fraudulent_given_options * P_options + P_fraudulent_given_crypto * P_crypto)
print(P_options_given_fraudulent)
# Q2
# we compute this by the law of total probability
P_fraudulent = P_fraudulent_given_equities * P_equities + P_fraudulent_given_options * P_options + P_fraudulent_given_crypto * P_crypto
print(P_fraudulent)
# Q3
# In the event space of at least one fraudulent, it can be portioned into three exclusive events (or sub space): 1) the first user is fraudulent and the second is not; 2) the first is not and the second is fraudulent; 3) both are fraudulent.
# We compute the probability for each event, that is P_fraudulent * (1 - P_fraudulent), (1 - P_fraudulent) * P_fraudulent which is equal to the preceding one, and P_fraudulent * P_fraudulent, and then obtain the final result.
P = (P_fraudulent * P_fraudulent) / (P_fraudulent * (1 - P_fraudulent) * 2 + P_fraudulent * P_fraudulent)
print(round(P, 4))



# logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# fit the model with data
target = np.copy(data_target)
target[data_target > 5] = 1
target[data_target <= 5] = 0

logreg.fit(data_feature,target)

#
X_test = data_feature[1:4, :]
y_pred = logreg.predict(X_test)
print(y_pred)

from sklearn import metrics
y_test = [1.0, 0.0, 1.0]
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

from sklearn.metrics import roc_auc_score
clf = LogisticRegression(solver="liblinear", random_state=0).fit(data_feature, target)
auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print('auc score', auc_score)

import numpy as np
import statsmodels.api as sm
def get_stats():
    results = sm.OLS(target, data_feature).fit()
    print(results.summary())
get_stats()