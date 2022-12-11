import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import graphviz


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import load_iris

train_set = pd.read_csv('modified_data/train.csv')
test_set = pd.read_csv('modified_data/test.csv')
gender_submission = pd.read_csv('modified_data/gender_submission.csv')
test_set['Survived'] = gender_submission['Survived']



train_set=train_set.drop_duplicates()

train_set=train_set.drop(['Name','Ticket','Cabin','Embarked','PassengerId'],axis=1)
test_set=test_set.drop(['Name','Ticket','Cabin','Embarked','PassengerId'],axis=1)


# https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
########################################################################################################################

trained,valid_set = train_test_split(train_set, test_size = 0.3, random_state = 7)
trained.info()
train_x = trained.drop(['Survived'], axis = 1)
train_y = pd.DataFrame(trained['Survived'])

valid_x = valid_set.drop(['Survived'], axis = 1)
valid_y = pd.DataFrame(valid_set['Survived'])

test_x = test_set.drop(['Survived'], axis = 1)
test_y = pd.DataFrame(test_set['Survived'])
##############################################################################
LGBClassifier = lgb.LGBMClassifier(objective='binary', max_depth = 8, learning_rate = 0.01, n_estimators = 9000, max_bin = 200, bagging_freq = 4, bagging_seed = 8, feature_fraction = 0.2, feature_fraction_seed = 8, min_sum_hessian_in_leaf = 11, verbose = -1, random_state = 42)
lgbm = LGBClassifier.fit(train_x.values, train_y.values.ravel(), eval_set = [(train_x.values, train_y), (valid_x.values, valid_y)],eval_metric ='logloss', early_stopping_rounds = 20, verbose =False)
feature_imp= pd.DataFrame(sorted(zip(lgbm.feature_importances_, test_x.columns), reverse = True), columns = ['Value', 'Feature'])

plt.figure(figsize=(7,5))
sb.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
plt.tight_layout()
plt.show()
########################################################################################################################
result_lst =[]
max_accuracy =0
val_y_prob = lgbm.predict_proba(
valid_x.values)[:, 1]

for n in range(0,60):
    threshold = round(((n+1)*0.01),2)
    pred_yn = val_y_prob.copy()
    pred_yn = np.where(pred_yn > threshold, 1., 0.)
    result_dict = {}
    precision, recall, f1_score, support = precision_recall_fscore_support(valid_y.values.ravel(), pred_yn, average='binary')
    accuracy = accuracy_score(valid_y.values.ravel(), pred_yn)
    kappa = cohen_kappa_score(valid_y.values.ravel(), pred_yn)
    result_dict ={'Threshold': threshold, 'Accuracy': round(accuracy,4), 'Precision': round(precision,4), 'Recall': round(recall,4), 'F1_Score': round(f1_score,4), 'Kappa': round(kappa,4)}
    result_lst.append(result_dict)

    if max_accuracy <= accuracy:
        max_accuracy = accuracy
        opt_threshold = threshold

    confMat = confusion_matrix(
valid_y.values.ravel(), pred_yn, labels=[1,0])

matric_df = pd.DataFrame(result_lst, columns=['Threshold','Accuracy', 'Precision', 'Recall', 'F1_Score', 'Kappa'])
matric_df.to_csv('REC_scores.csv',sep=',', header=True, index=False, encoding='UTF-8')

print('Threshold=%f'%(opt_threshold))
########################################################################################################################
predict_lgbm = lgbm.predict_proba(train_x.values)[:,1]
pred_train = np.where(predict_lgbm > opt_threshold, 1., 0.)

tp, fn, fp, tn = confusion_matrix(train_y.values.ravel(), pred_train, labels=[1,0]).ravel()
########################################################################################################################
# Confusion Matrix
conf_matrix = pd.DataFrame(confusion_matrix(train_y.values.ravel(), pred_train), columns=['Predicted Value 0', 'Predicted Value 1'], index=['True Value 0', 'True Value 1'])

print("Confusion Matrix")
print(conf_matrix.T)
print("\nClassification Report: Train")
print(classification_report(
train_y.values.ravel(), pred_train))
########################################################################################################################
# Roc Curve
# Training Set
fpr, tpr, _ = roc_curve(train_y.values.ravel(), predict_lgbm)
roc_auc = auc(fpr, tpr)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, color='red')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

Accuracy_Rate = (tp + tn) / (tp + tn + fp + fn)
Recall_Rate = tp / (tp + fn)
Precision_Rate = tp / (tp + fp)
Specificity_Rate = tn / (tn + fp)
F1_Score = (Precision_Rate * Recall_Rate) / (Precision_Rate + Recall_Rate) * 2

print("Summary")
print(" - Accuracy Rate    : {:2.3f} %".format(Accuracy_Rate*100))
print(" - Recall Rate      : {:2.3f} %".format(Recall_Rate*100))
print(" - Precision Rate   : {:2.3f} %".format(Precision_Rate*100))
print(" - Specificity Rate : {:2.3f} %".format(Specificity_Rate*100))
print(" - F1 Score         : {:2.3f} ".format(F1_Score*100))
print(" - ROC AUC          : {:2.3f} %".format(roc_auc*100))



########################################################################################################################
# valid_set
predict_lgbm = lgbm.predict_proba(valid_x.values)[:,1]
pred_val = np.where(predict_lgbm > opt_threshold, 1., 0.)

tp, fn, fp, tn = confusion_matrix(valid_y.values.ravel(), pred_val, labels=[1,0]).ravel()

conf_matrix = pd.DataFrame(confusion_matrix(valid_y.values.ravel(), pred_val), columns=['Predicted Value 0', 'Predicted Value 1'], index=['True Value 0', 'True Value 1'])

print("Confusion Matrix")
print(conf_matrix.T)
print("\nClassification Report: Validation")
print(classification_report(
train_y.values.ravel(), pred_train))

fpr, tpr, _ = roc_curve(valid_y.values.ravel(), predict_lgbm)
roc_auc = auc(fpr, tpr)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

Accuracy_Rate = (tp + tn) / (tp + tn + fp + fn)
Recall_Rate = tp / (tp + fn)
Precision_Rate = tp / (tp + fp)
Specificity_Rate = tn / (tn + fp)
F1_Score = (Precision_Rate * Recall_Rate) / (Precision_Rate + Recall_Rate) * 2

print("Summary")
print(" - Accuracy Rate    : {:2.3f} %".format(Accuracy_Rate*100))
print(" - Recall Rate      : {:2.3f} %".format(Recall_Rate*100))
print(" - Precision Rate   : {:2.3f} %".format(Precision_Rate*100))
print(" - Specificity Rate : {:2.3f} %".format(Specificity_Rate*100))
print(" - F1 Score         : {:2.3f}".format(F1_Score*100))
print(" - ROC AUC          : {:2.3f} %".format(roc_auc*100))

########################################################################################################################
# Test
predict_lgbm = lgbm.predict_proba(test_x.values)[:,1]
pred_test = np.where(predict_lgbm > opt_threshold, 1., 0.)

tp, fn, fp, tn = confusion_matrix(test_y.values.ravel(), pred_test, labels=[1,0]).ravel()

conf_matrix = pd.DataFrame(confusion_matrix(test_y.values.ravel(), pred_test), columns=['Predicted Value 0', 'Predicted Value 1'], index=['True Value 0', 'True Value 1'])

print("Confusion Matrix")
print(conf_matrix.T)
print("\nClassification Report: Test")
print(classification_report(
train_y.values.ravel(), pred_train))

fpr, tpr, _ = roc_curve(test_y.values.ravel(), predict_lgbm)
roc_auc = auc(fpr, tpr)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, color='green')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

Accuracy_Rate = (tp + tn) / (tp + tn + fp + fn)
Recall_Rate = tp / (tp + fn)
Precision_Rate = tp / (tp + fp)
Specificity_Rate = tn / (tn + fp)
F1_Score = (Precision_Rate * Recall_Rate) / (Precision_Rate + Recall_Rate) * 2

print("Summary")
print(" - Accuracy Rate    : {:2.3f} %".format(Accuracy_Rate*100))
print(" - Recall Rate      : {:2.3f} %".format(Recall_Rate*100))
print(" - Precision Rate   : {:2.3f} %".format(Precision_Rate*100))
print(" - Specificity Rate : {:2.3f} %".format(Specificity_Rate*100))
print(" - F1 Score         : {:2.3f} ".format(F1_Score*100))
print(" - ROC AUC          : {:2.3f} %".format(roc_auc*100))


########################################################################################################################