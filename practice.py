import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = fetch_openml('adult', version=2, as_frame=True)
df: pd.DataFrame = data.frame
target = 'class'  # target column name

# PART 1 - EXPLORATORY DATA ANALYSIS
# 1.)
print(df.info(), end='\n\n')

# 2.)
class_counts = df[target].value_counts()
plt.title('Imbalance Ratio of Target Distribution', weight='bold')
plt.bar(class_counts.index, class_counts, edgecolor='black', label=f'IR: {class_counts.max()/class_counts.min()*100:.2f}%')
plt.legend()
plt.show()

# 3.)
num_features = df.select_dtypes(include='number')
num_features.hist(grid=False, edgecolor='black')
plt.suptitle('Numeric Features Histograms', weight='bold')
plt.tight_layout()
plt.show()

# 4.)
cat_features = df.select_dtypes(include='category')
cat_features.drop(labels='class', axis=1, inplace=True)
plt.figure(figsize=(12, 6))
plt.suptitle('Categorical Features', weight='bold')

for i, column in enumerate(cat_features.columns, start=1):
    plt.subplot(2, 4, i)
    bars = sns.countplot(data=cat_features, y=column, edgecolor='black', order=cat_features[column].value_counts().index[:5])
    bars.bar_label(bars.containers[0], labels=cat_features[column].value_counts().iloc[:5])
    bars.spines['top'].set_visible(False)
    bars.spines['right'].set_visible(False)
    plt.ylabel(column, weight='bold')

plt.tight_layout()
plt.show()

# 5.)
sns.heatmap(num_features.corr(), cmap='coolwarm')
plt.title('Heatmap Correlation', weight='bold')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 6.)
numf_with_target = num_features.copy()
numf_with_target['class'] = df['class']
plt.figure(figsize=(9,6))
plt.suptitle('Target vs. Features', weight='bold')

for i, column in enumerate(numf_with_target.columns[:-1], start=1):
    plt.subplot(2, 3, i)
    sns.boxplot(numf_with_target, x='class', y=column)
    plt.ylabel(column, weight='bold')

plt.tight_layout()
plt.show()

#============================================#

# PART 2 - DATA PRE-PROCESSING
# 1.)
num_imputer = SimpleImputer(strategy='median')
num_imputer.set_output(transform='pandas')
df[num_features.columns] = num_imputer.fit_transform(num_features)

cat_imputer = SimpleImputer(strategy='most_frequent')
cat_imputer.set_output(transform='pandas')
df[cat_features.columns] = cat_imputer.fit_transform(cat_features)

# 2.)
nominal_cats = cat_features.columns.to_list()
encoder = OneHotEncoder(sparse_output=False)
encoder.set_output(transform='pandas')
encoded_nominalcats = encoder.fit_transform(df[nominal_cats])
print('*ENCODED CATEGORICAL FEATURES\n', encoded_nominalcats, end='\n\n')

# 3.)
scaler = StandardScaler()
scaler.set_output(transform='pandas')
scaled_numf = scaler.fit_transform(num_features)
print('*SCALED NUM FEATURES\n', scaled_numf, end='\n\n')

# 4.)
df_mod: pd.DataFrame = pd.concat([scaled_numf, encoded_nominalcats], axis=1)
df_mod[target] = df[target].map({'<=50K':0, '>50K':1})
X = df_mod[df_mod.columns[:-1]]
y = df_mod[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

#============================================#

# PART 3 - BUILDING BASELINE MODELS
# 1.)
logreg = LogisticRegression(max_iter=500)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_cv_results = cross_validate(estimator=logreg, X=X_train, y=y_train, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], cv=skfold, n_jobs=4)
cv_res_df = pd.DataFrame({'log_results':[log_cv_results['test_accuracy'].mean().round(2), log_cv_results['test_precision'].mean().round(2), log_cv_results['test_recall'].mean().round(2), log_cv_results['test_f1'].mean().round(2), log_cv_results['test_roc_auc'].mean().round(2)]}, index=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

# 2.)
rf_clf = RandomForestClassifier(random_state=42, n_jobs=4)
rf_cv_results = cross_validate(estimator=rf_clf, X=X_train, y=y_train, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], cv=skfold, n_jobs=4)
cv_res_df['rf_results'] = [rf_cv_results['test_accuracy'].mean().round(2), rf_cv_results['test_precision'].mean().round(2), rf_cv_results['test_recall'].mean().round(2), rf_cv_results['test_f1'].mean().round(2), rf_cv_results['test_roc_auc'].mean().round(2)]
print('*BASELINE CLF CV RESULTS:\n', cv_res_df, end='\n\n')

# 3.)
print('Best Performing Classifier:', rf_clf, end='\n\n')

#============================================#

# PART 4 - FEATURE ENGINEERING
# 1.)
def age_grouping(x):
    if x < 39:
        return 'Young'
    elif x < 59:
        return 'Middle'
    else:
        return 'Senior'

df_mod['age_group'] = df['age'].map(age_grouping)
ag_encoded = encoder.fit_transform(df_mod['age_group'].to_frame())
df_mod[ag_encoded.columns] = ag_encoded
df_mod.drop('age_group', axis=1, inplace=True)
df_mod['capital_gain_ratio'] = df['capital-gain'] / df['hours-per-week']
df_mod[target] = df_mod.pop(target)

# 2.)
X_train = pd.concat([X_train, df_mod.loc[X_train.index, ag_encoded.columns], df_mod.loc[X_train.index, 'capital_gain_ratio']], axis=1)
X_test = pd.concat([X_test, df_mod.loc[X_test.index, ag_encoded.columns], df_mod.loc[X_test.index, 'capital_gain_ratio']], axis=1)

rf_clf.fit(X_train, y_train)
topf_df = pd.DataFrame({'features': df_mod.columns[:-1], 'feature_score':rf_clf.feature_importances_})
topf_df.sort_values(by='feature_score', ascending=False, inplace=True)
topf_df.drop(index=topf_df.index[10:], inplace=True)
topf_df.index = range(1, 11)
topf_df.index.name = 'rank'
print(topf_df, end='\n\n')

log_cv_results = cross_validate(estimator=logreg, X=X_train[topf_df['features'].to_list()], y=y_train, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], cv=skfold, n_jobs=4)
rf_cv_results = cross_validate(estimator=rf_clf, X=X_train[topf_df['features'].to_list()], y=y_train, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], cv=skfold, n_jobs=4)
cv_res_df = pd.DataFrame({'log_results':[log_cv_results['test_accuracy'].mean().round(2), log_cv_results['test_precision'].mean().round(2), log_cv_results['test_recall'].mean().round(2), log_cv_results['test_f1'].mean().round(2), log_cv_results['test_roc_auc'].mean().round(2)], 
    'rf_results':[rf_cv_results['test_accuracy'].mean().round(2), rf_cv_results['test_precision'].mean().round(2), rf_cv_results['test_recall'].mean().round(2), rf_cv_results['test_f1'].mean().round(2), rf_cv_results['test_roc_auc'].mean().round(2)]}, index=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

print('*NEW CLF CV RESULTS:\n', cv_res_df, end='\n\n')

#============================================#

# PART 5 - HYPERPARAMETER
# 1.)
params = {'n_estimators':[100, 200, 300, 400, 500], 'max_depth':[5, 10, 20, 30], 'min_samples_split':[5, 10, 20, 30]}
rsearch_cv = RandomizedSearchCV(estimator=rf_clf, param_distributions=params, n_iter=15, scoring='roc_auc', n_jobs=4, cv=skfold)
rsearch_cv.fit(X_train, y_train)

# 2.)
print('*Best Hyperparameters for RandomForestClassifier:\n', rsearch_cv.best_params_, end='\n\n')
print('*ROC_AUC CV SCORE: ', rsearch_cv.best_score_, end='\n\n')

# 3.)
tuned_rf_clf = RandomForestClassifier(random_state=42, n_jobs=4)
tuned_rf_clf.set_params(**rsearch_cv.best_params_)
tuned_rf_clf.fit(X_train, y_train)

#============================================#

# PART 6: FINAL EVALUATION ON TEST SET
# 1.)
final_acc = accuracy_score(y_true=y_test, y_pred=tuned_rf_clf.predict(X_test))
final_prec = precision_score(y_true=y_test, y_pred=tuned_rf_clf.predict(X_test))
final_rec = recall_score(y_true=y_test, y_pred=tuned_rf_clf.predict(X_test))
final_f1 = f1_score(y_true=y_test, y_pred=tuned_rf_clf.predict(X_test))
final_roc_auc = roc_auc_score(y_true=y_test, y_score=tuned_rf_clf.predict_proba(X_test)[:, 1])

print('*FINAL SCORES:')
print('Accuracy: ', final_acc,
    '\nPrecision: ', final_prec,
    '\nRecall: ', final_rec,
    '\nF1: ', final_f1,
    '\nROC_AUC: ', final_roc_auc, end='\n\n')

sns.heatmap(confusion_matrix(y_true=y_test, y_pred=tuned_rf_clf.predict(X_test)), cmap='coolwarm', annot=[['TN', 'FN'],['FP', 'TP']], fmt='')
plt.title('Final Confusion Matrix', weight='bold')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

RocCurveDisplay.from_estimator(estimator=tuned_rf_clf, X=X_test, y=y_test)
plt.title('Final ROC_AUC Curve', weight='bold')
plt.plot([0,1], [0,1], linestyle='--', color='red', label='Random Guessing (AUC = 0.50)')
plt.xlabel('FPR', weight='bold')
plt.ylabel('TPR', weight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# 2.)
'''
*Which metric is appropriate for this problem?
 - I think f1 and roc_auc metrics are appropriate for this problem because the data is imbalance,
so in order to evaluate it I will need f1 metric, and roc_auc metric to know if the model is good even it is imbalance.

*What does the confusion matrix tell you about the model's strengths and weaknesses?
 - It tells me that it is good on getting true negatives than the positives and it make sense because the data is imbalance, 
for false negatives and positives it only has small percentage of them.
'''

#============================================#

# PART 7: BUILD A COMPLETE PIPELINE
# 1.)
Xf_train = df.loc[X_train.index]
Xf_test = df.loc[X_test.index]

cat_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
num_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

column_transformer = ColumnTransformer(transformers=[('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num_columns),
                    ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(sparse_output=False)), cat_columns)], remainder='drop')
column_transformer.set_output(transform='pandas')

complete_model = make_pipeline(column_transformer, tuned_rf_clf)
complete_model.fit(Xf_train, y_train)

# 2.)
finalpipe_acc = accuracy_score(y_true=y_test, y_pred=complete_model.predict(Xf_test))
finalpipe_prec = precision_score(y_true=y_test, y_pred=complete_model.predict(Xf_test))
finalpipe_rec = recall_score(y_true=y_test, y_pred=complete_model.predict(Xf_test))
finalpipe_f1 = f1_score(y_true=y_test, y_pred=complete_model.predict(Xf_test))
finalpipe_roc_auc = roc_auc_score(y_true=y_test, y_score=complete_model.predict_proba(Xf_test)[:, 1])

print('*FINAL PIPELINE SCORES:')
print('Accuracy: ', finalpipe_acc,
    '\nPrecision: ', finalpipe_prec,
    '\nRecall: ', finalpipe_rec,
    '\nF1: ', finalpipe_f1,
    '\nROC_AUC: ', finalpipe_roc_auc, end='\n\n')

#============================================#

# PART 8 - MODEL INTERPRETATION
# 1.
preprocessor = complete_model.named_steps['columntransformer']
transformed_data = preprocessor.transform(Xf_train)
topf_df = pd.DataFrame({'features':transformed_data.columns, 'feature_score':complete_model[-1].feature_importances_.round(6)})
topf_df.sort_values(by='feature_score', ascending=False, inplace=True)
topf_df.drop(index=topf_df.index[10:], inplace=True)
topf_df.index.name = 'rank'
topf_df.index = range(1, 11)
print(topf_df)