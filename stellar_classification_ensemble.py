#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:31:45 2023

@author: georg


Inhalt des Datensatzes:
    Der Datensatz beinhaltet 100000 Objekte, die in drei Klassen (Stern, Galaxie,
    Quasar) einbeteilt werden, diese Einteilung geschieht aufgrund seines 
    elektro-magnetischen Spektrums. Dieses Spektrum ist in diesem Datensatz insofern repräsentiert 
    

Struktur des Datensatzes

obj_ID = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS

Verortung am Himmel:
    alpha = Right Ascension angle (at J2000 epoch)
    delta = Declination angle (at J2000 epoch)

Relative Helligkeit mit einem jeweiligen Filter, also in diesem Spektrum:
    u = Ultraviolet filter in the photometric system
    g = Green filter in the photometric system
    r = Red filter in the photometric system
    i = Near Infrared filter in the photometric system
    z = Infrared filter in the photometric system

Daten über die Scans
    run_ID = Run Number used to identify the specific scan
    rereun_ID = Rerun Number to specify how the image was processed
    cam_col = Camera column to identify the scanline within the run
    field_ID = Field number to identify each field

Klassifizierung:
    spec_obj_ID = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)
    class = object class (galaxy, star or quasar object)

Rotverschiebung aufgrund der Entfernung und der Eigenbewegung    
    redshift = redshift value based on the increase in wavelength

Daten zu den Beobachtungsumständen
    plate = plate ID, identifies each plate in SDSS
    MJD = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
    fiber_ID = fiber ID that identifies the fiber that pointed the light at the focal plane in each observation
"""




# %% importe
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin


from catboost import CatBoostClassifier
import xgboost as xgb

from hyperopt import tpe, hp, fmin, Trials
# from hyperopt.pyll.base import scope

# %% Einlesen des Datensatzes
raw_df = pd.read_csv("star_classification.csv")

print(raw_df.head())

# %% bereinigen

# relevante Features:
# rel_features = ["u", "g","r", "i", "z", "class"]    
rel_features = ["u", "g","r", "i", "z", "redshift", "class"]    



df = raw_df.loc[:, rel_features]

# Schauen, ob NaN-Werte da sind
nan_count = df.isna().sum()
print(nan_count)

print(df.describe())

# Ausreißer entfernen
outlier = df[df['u']==-9999.000].index
df.drop(outlier,inplace = True)
print(df.describe())
a = df.describe()


# %% Visuaiisierung der Daten

# Verteilung der Klassen anschauen
print(df["class"].value_counts())

# class
# GALAXY    59445
# STAR      21593
# QSO       18961

# %% Histogramme

fig, axes = plt.subplots(2, 3, figsize=(10, 8))

for i, f in enumerate(rel_features[:-2]):
    gal_dat = df[df["class"] == "GALAXY"][f]
    star_dat = df[df["class"] == "STAR"][f]
    qso_dat = df[df["class"] == "QSO"][f]
    
    # Genialer Trick zum Durchlaufen von Zeilen und Spalten
    ax = axes[i // 3, i % 3]  
    
    # Histogramme erstellen
    ax.hist(gal_dat, bins=50, alpha=0.5, label="Galaxy")
    ax.hist(star_dat, bins=50, alpha=0.5, label="Star")
    ax.hist(qso_dat, bins=50, alpha=0.5, label="Quasar")
    
    # Diagramme beschriften
    ax.set_xlabel(f"Helligkeit ({f})")
    ax.set_ylabel("Anzahl")
      
    # Legende
    ax.legend()

# Titel
fig.suptitle("Histogramme der scheinbaren Helligkeiten")
# Layout anpassen
plt.tight_layout()

# Abspeichern
filename = "Images/Histogramme"
plt.savefig(filename, dpi=600)

# Anzeigen
plt.show()

# %% Redshift
gal_dat = df[df["class"] == "GALAXY"]["redshift"]
star_dat= df[df["class"] == "STAR"]["redshift"]
qso_dat = df[df["class"] == "QSO"]["redshift"]
plt.figure(dpi=600)

# Histogramme erstellen
plt.hist(gal_dat, bins=50, alpha=0.5, label="Galaxy")
plt.hist(star_dat, bins=50, alpha=0.5, label="Star")
plt.hist(qso_dat, bins=50, alpha=0.5, label="Quasar")

# Diagramme beschriften
plt.xlabel("Rotverschiebung in z")
plt.ylabel("Anzahl")
plt.title("Histogramm der Rotverschiebung")

# Legende
plt.legend()

# Abspeichern
filename = "Images/Rotverschiebung"
plt.savefig(filename)
# Anzeigen
plt.show()




# %% Aufteilen, Skalieren, Numerisieren und Splitten

X = df.drop(["class"], axis=1)
y = df["class"]


# Mapping für Klassen
class_mapping = {'STAR': 0, 'GALAXY': 1, 'QSO': 2}

# Mapping anwenden
y = y.map(class_mapping)

# Split
# Stratify sorgt für die gleiche Verteilung der Klassen in Train und Test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, 
                                                    stratify = y)


# Scaling der Daten - für alle Fälle
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %% Cat Boost

cat_model = CatBoostClassifier(
    iterations = 1000,
    loss_function='MultiClass', 
    bootstrap_type = "Bayesian", # Stichprobenerzeugung aus den Trainingsdaten
    eval_metric = 'MultiClass', # Evaluierungsmetrik
    leaf_estimation_iterations = 50, # quasi Epochen innerhalb eines Baumes/Iteration
    random_strength = 0.5, # Wie stark sollen die Bäume sich unterscheiden
    depth = 7, # Tiefe der Bäume
    l2_leaf_reg = 5, # Regularisierung dr Blätter; je höher, desto geringer die Komplexität des Modells
    learning_rate=0.1, 
    bagging_temperature = 0.5, # Je höher, desto eher Anpassung an Trainingsdaten
    task_type = "GPU",
)

# %%
# training the model
cat_model.fit(X_train,y_train)

# %% Score

y_pred_cat = cat_model.predict(X_test)

acc_cat = cat_model.score(X_test,y_test)

print("accuracy of the catboost: ", cat_model.score(X_test,y_test))

# 0.97765


# confusion metrics of the LightGBM and plotting the same
confusion_matrix_LightGBM = confusion_matrix(y_test,y_pred_cat)
print(confusion_matrix_LightGBM) 

print(classification_report(y_test,y_pred_cat))

# Mit Redshift
# [[ 4294    24     1]
#  [   42 11721   126]
#  [    0   254  3538]]
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      4319
#            1       0.98      0.99      0.98     11889
#            2       0.97      0.93      0.95      3792

#     accuracy                           0.98     20000
#    macro avg       0.98      0.97      0.97     20000
# weighted avg       0.98      0.98      0.98     20000




# Ohne Redshift
# accuracy of the catboost:  0.8695
# [[ 3144   775   400]
#  [  362 11201   326]
#  [  345   402  3045]]
#               precision    recall  f1-score   support

#            0       0.82      0.73      0.77      4319
#            1       0.90      0.94      0.92     11889
#            2       0.81      0.80      0.81      3792

#     accuracy                           0.87     20000
#    macro avg       0.84      0.82      0.83     20000
# weighted avg       0.87      0.87      0.87     20000



# %%

# Ermittle die Feature-Bedeutungen in Prozent
feature_importances_cat = cat_model.feature_importances_


# # Gib die Feature-Bedeutungen aus
for feature_name, importance in zip(X.columns, feature_importances_cat):
    print(f"Feature: {feature_name}, Importance: {importance}")

feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances_cat)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance - CatBoost')
# Abspeichern
filename = "Images/CatBoost_Feature"
plt.savefig(filename, dpi=600)
# Anzeigen
plt.show()



# Feature: u, Importance: 14.35764507775051
# Feature: g, Importance: 11.94906850081733
# Feature: r, Importance: 6.801895013516247
# Feature: i, Importance: 7.933711756466816
# Feature: z, Importance: 10.069895451245767
# Feature: redshift, Importance: 48.88778420020333




# %% XGBoost

xgb_model = xgb.XGBClassifier(learning_rate=0.001, 
                              max_depth=5, 
                              n_estimators=100)


# %% Fit
xgb_model.fit(X_train, y_train)

# %%  Score


print(xgb_model.score(X_test, y_test))

y_pred_xgb = xgb_model.predict(X_test)
cm_xgb = confusion_matrix(y_test,y_pred_xgb) 
print(cm_xgb)
print(classification_report(y_test,y_pred_xgb))

# 0.9714

# [[ 4317     2     0]
#  [   19 11733   137]
#  [    0   414  3378]]
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00      4319
#            1       0.97      0.99      0.98     11889
#            2       0.96      0.89      0.92      3792

#     accuracy                           0.97     20000
#    macro avg       0.97      0.96      0.97     20000
# weighted avg       0.97      0.97      0.97     20000


# %% Features
# Ermittle die Feature-Bedeutungen
feature_importances_xgb = xgb_model.feature_importances_

feature_names = X.columns

# Erstellen des Plots
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances_xgb)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance - XGBoost')
# Abspeichern
filename = "Images/XGBoost_Feature"
plt.savefig(filename, dpi=600)
plt.show()


for feature_name, importance in zip(X.columns, feature_importances_xgb):
    print(f"Feature: {feature_name}, Importance: {importance}")


# Feature: u, Importance: 0.033222347497940063
# Feature: g, Importance: 0.0729372575879097
# Feature: r, Importance: 0.005890898406505585
# Feature: i, Importance: 0.008232079446315765
# Feature: z, Importance: 0.04086834192276001
# Feature: redshift, Importance: 0.8388490676879883


# %% AdaBoost

dtree = DecisionTreeClassifier()
cl_ada = AdaBoostClassifier(n_estimators=100, 
                            estimator = dtree, 
                            learning_rate=1) 


# %% Fit

cl_ada.fit(X_train,y_train)


# %% Score

acc_ada = cl_ada.score(X_test, y_test)

print(cl_ada.score(X_test, y_test))

# Confusion Matrix
y_pred_ada = cl_ada.predict(X_test)
cm = confusion_matrix(y_test,y_pred_ada) 
print(cm)

# Classification Report
print(classification_report(y_test,y_pred_ada))

# 0.9656

# [[ 4286    33     0]
#  [   28 11535   326]
#  [    0   301  3491]]
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      4319
#            1       0.97      0.97      0.97     11889
#            2       0.91      0.92      0.92      3792

#     accuracy                           0.97     20000
#    macro avg       0.96      0.96      0.96     20000
# weighted avg       0.97      0.97      0.97     20000



# %% Features

feature_importances_ada = cl_ada.feature_importances_


for feature_name, importance in zip(X.columns, feature_importances_ada):
    print(f"Feature: {feature_name}, Importance: {importance}")


feature_names = X.columns

# Erstellen des Plots
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances_ada)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance - AdaBoost')
# Abspeichern
filename = "Images/AdaBoost_Feature"
plt.savefig(filename, dpi=600)
plt.show()


# Feature: u, Importance: 0.023484741883020296
# Feature: g, Importance: 0.05078080474034423
# Feature: r, Importance: 0.01230008498938542
# Feature: i, Importance: 0.015396578403756075
# Feature: z, Importance: 0.016761031496711987
# Feature: redshift, Importance: 0.8812767584867821


# %% Hyperopt CatBoost

# Search Space
space = {
    'iterations': hp.choice('iterations', [500, 1000, 1500]),
    'leaf_estimation_iterations': hp.choice('leaf_estimation_iterations', [30, 50, 100]),
    'random_strength': hp.uniform('random_strength', 0.1, 1.0),
    'depth': hp.choice('depth', [3, 5, 7, 9]),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),  # 1e-5 bis 1
    'bagging_temperature': hp.uniform('bagging_temperature', 0.1, 1.0),
}


# Objective Function
def catboost_objective(params):
    model = CatBoostClassifier(
        iterations=params['iterations'],
        loss_function='MultiClass',
        bootstrap_type = "Bayesian",
        eval_metric='MultiClass',
        leaf_estimation_iterations=params['leaf_estimation_iterations'],
        random_strength=params['random_strength'],
        depth=params['depth'],
        l2_leaf_reg=params['l2_leaf_reg'],
        learning_rate=params['learning_rate'],
        bagging_temperature=params['bagging_temperature'],
        task_type="GPU",
        early_stopping_rounds=100,  # Anzahl der Runden ohne Verbesserung beim ersten Versuch 50
        verbose=25,  # Alle n Iterationen kommt die Ausgabe des Trainingsprozesses
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test)
    )

    best_iteration = model.best_iteration_  # Beste Iteration basierend auf dem Early Stopping

    y_pred = model.predict(X_test)  # Mache Vorhersagen auf Testdaten
    accuracy = accuracy_score(y_test, y_pred)  # Berechne die Genauigkeit

    return -accuracy  # Minimiere die negative Genauigkeit (Hyperopt maximiert das Ziel)

# %% Durchführung der Optimierung

trials_cat = Trials()
best_cat = fmin(
    fn=catboost_objective,
    space=space,
    algo=tpe.suggest, # sucht aktiv nach besseren Kombinationen
    max_evals=50,
    trials=trials_cat
)

print("Beste Hyperparameter-Kombination:", best_cat)

# =============================================================================
# 100%|██████████| 50/50 [34:24<00:00, 41.29s/trial, best loss: -0.97725]
# Beste Hyperparameter-Kombination: 
#     {'bagging_temperature': 0.2902181183828039, 
#       'depth': 3, 
#       'iterations': 0, 
#       'l2_leaf_reg': 8.069867859500645, 
#       'leaf_estimation_iterations': 0, 
#       'learning_rate': 0.12033950249399486, 
#       'random_strength': 0.3235686332225774}
# =============================================================================

# =============================================================================
# 
# 100%|██████████| 50/50 [1:07:25<00:00, 80.91s/trial, best loss: -0.97505]
# Beste Hyperparameter-Kombination: 
#     {'bagging_temperature': 0.21523723549780033, 
#      'depth': 3, 
#      'iterations': 1, 
#      'l2_leaf_reg': 3.7756771292461475, 
#      'leaf_estimation_iterations': 1, 
#      'learning_rate': 0.06606169824403287, 
#      'random_strength': 0.6992133651188215}
# =============================================================================


# %% Hyperopt XGBoost

# Search Space
space = {
    'learning_rate': hp.loguniform('learning_rate', -6, 0),  # 1e-6 bis 1
    'max_depth': hp.choice('max_depth', range(1, 10)),
    'n_estimators': hp.choice('n_estimators', range(50, 200))
}

# Objective Function
def xgb_objective(params):
    model = xgb.XGBClassifier(
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        early_stopping_rounds=100
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    score = model.score(X_test, y_test)
    return -score  # Negative Wertung, da Hyperopt nach dem Minimum sucht

# %%
# Optimierung
trials_xgb = Trials()
best_xgb = fmin(fn=xgb_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials_xgb)

# Drucke die besten Hyperparameter
print("Beste Hyperparameter:")
print(best_xgb)

# =============================================================================
# Ergebnis mit early_stopping 100
# 100%|██████████| 100/100 [14:56<00:00,  8.97s/trial, best loss: -0.9787]
# Beste Hyperparameter:
# {'learning_rate': 0.030073928445922528, 
#  'max_depth': 7, 
#  'n_estimators': 117}
# =============================================================================


# =============================================================================
# 100%|██████████| 100/100 [14:58<00:00,  8.98s/trial, best loss: -0.97965]
# Beste Hyperparameter:
# {'learning_rate': 0.17353909349509708, 
#  'max_depth': 6, 
#  'n_estimators': 114}
# =============================================================================


# %% Hyperopt Ada

# Search Space
space = {
    'n_estimators': hp.choice('n_estimators', range(20, 200)),
    'learning_rate': hp.loguniform('learning_rate', -5, 0)
}

# Objective-Function
def ada_objective(params):
    dtree = DecisionTreeClassifier()
    cl_ada = AdaBoostClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        estimator=dtree
    )
    cl_ada.fit(X_train, y_train)
    y_pred = cl_ada.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return -score  # Negative Wertung, da Hyperopt nach dem Minimum sucht


# %% Optimierung

trials_ada = Trials()

best_ada = fmin(fn=ada_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=200,
                trials=trials_ada)


# Drucke die besten Hyperparameter
print("Beste Hyperparameter:")
print(best_ada)

# =============================================================================
# 100%|██████████| 200/200 [02:38<00:00,  1.26trial/s, best loss: -0.9635]
# Beste Hyperparameter:
# {'learning_rate': 0.03357030269724003, 
#  'n_estimators': 152}
# =============================================================================

# =============================================================================
# 100%|██████████| 200/200 [02:50<00:00,  1.17trial/s, best loss: -0.96725]
# Beste Hyperparameter:
# {'learning_rate': 0.023562663925221328, 
#  'n_estimators': 51}
# =============================================================================

# %% Voting mit allen drei

# %%

# =============================================================================
# Die Voting Funktion
# =============================================================================

def voting(arr1, arr2, arr3, acc1, acc2, acc3):
    output = np.zeros_like(arr1)  # Numpy Array mit der gleichen Form wie arr1
    
    accuracies = [acc1, acc2, acc3]
    
    for i, (a1, a2, a3) in enumerate(zip(arr1, arr2, arr3)):
        if a1 == a2 == a3:
            output[i] = a1
        elif a1 == a2 and a1 != a3:
            output[i] = a1
        elif a1 == a3 and a1 != a2:
            output[i] = a1
        elif a2 == a3 and a2 != a1:
            output[i] = a2
        else:
            max_acc = max(accuracies)
            max_index = accuracies.index(max_acc)
            output[i] = [a1, a2, a3][max_index]
    
    return output



# %% Vorarbeiten


# Der Reshape für CatBoost
y_pred_cat =  y_pred_cat.reshape(-1)


# xgb mit neuen  Parametern
xgb_model_hyp = xgb.XGBClassifier(learning_rate=0.1735, 
                              max_depth=6, 
                              n_estimators=114)

# xgb_model_hyp muss gefittet werden
xgb_model_hyp.fit(X_train, y_train)

# xgb_model_hyp predicion
y_pred_xgb_hyp = xgb_model_hyp.predict(X_test)

acc_xgb_hyp = xgb_model_hyp.score(X_test, y_test)
print(acc_xgb_hyp)


cm_xgb = confusion_matrix(y_test,y_pred_xgb_hyp) 
print(cm_xgb)
print(classification_report(y_test,y_pred_xgb_hyp))



# %%
# Voting

# Initialisieren des VotingClassifiers mit den Basismodellen
voting_model = VotingClassifier(estimators=[('cat', cat_model), ('xgb', xgb_model_hyp), ('ada', cl_ada)],
                                voting='soft', weights=[2,1,1])

# Trainieren des VotingClassifiers
voting_model.fit(X_train, y_train)

# Vorhersagen mit dem VotingClassifiers
y_pred_vote_model = voting_model.predict(X_test)


# %% Score

acc_vote_model = np.mean(y_pred_vote_model == y_test)
print(acc_vote_model)



# confusion metrics of the LightGBM and plotting the same
confusion_matrix_LightGBM = confusion_matrix(y_test,y_pred_vote_model)
print(confusion_matrix_LightGBM) 

print(classification_report(y_test,y_pred_vote_model))



# %% Eigentliches Voting

y_pred_vote = voting(y_pred_ada, y_pred_cat, y_pred_xgb_hyp,
                     acc_ada, acc_cat, acc_xgb_hyp )

# %% Score

acc_vote = np.mean(y_pred_vote == y_test)
print(acc_vote)



# confusion metrics of the LightGBM and plotting the same
confusion_matrix_LightGBM = confusion_matrix(y_test,y_pred_vote)
print(confusion_matrix_LightGBM) 

print(classification_report(y_test,y_pred_vote))
