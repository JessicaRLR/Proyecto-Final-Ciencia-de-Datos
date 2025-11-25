# ProyectoCdD_Final.py
# Análisis de datos de expresión génica para clasificación de tipos de cáncer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import seaborn as sns
import os
plt.ion()  # modo interactivo para plots
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------
# Carga y limpieza de datos
# ---------------------------
expr_path = r"C:/Users/jessi/OneDrive/Documentos/Proyecto_CdD/EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2-v2.geneExp.tsv"
meta_path = r"C:/Users/jessi/OneDrive/Documentos/Proyecto_CdD/merged_sample_quality_annotations.tsv"

expr = pd.read_csv(expr_path, sep="\t", index_col=0)
meta = pd.read_csv(meta_path, sep="\t")
print(expr.head())
# Transponer
expr_T = expr.T

# Merge usando aliquot_barcode
merged = expr_T.merge(meta, left_index=True, right_on="aliquot_barcode", how="left")

# Filtrado de calidad: quitar muestras marcadas Do_not_use o excluidas por patología
if "Do_not_use" in merged.columns:
    # Normalizamos posibles valores booleanos/strings
    mask_do_not_use = merged["Do_not_use"].astype(str).str.lower().isin(["1","true","yes","y","t"])
    merged = merged[~mask_do_not_use].copy()

if "AWG_excluded_because_of_pathology" in merged.columns:
    mask_excluded = merged["AWG_excluded_because_of_pathology"].astype(str).str.lower().isin(["1","true","yes","y","t"])
    merged = merged[~mask_excluded].copy()

# Variable objetivo y features
target_col = "cancer type"   # tal como en el metadata

# Guardamos la etiqueta antes de quitar columnas metadata
y = merged[target_col].astype(str)

# Columnas metadata que no queremos en X
meta_cols = [
    "patient_barcode", "aliquot_barcode", "cancer type",
    "platform", "patient_annotation", "sample_annotation",
    "aliquot_annotation", "aliquot_annotation_updated",
    "AWG_excluded_because_of_pathology", "AWG_pathology_exclusion_reason",
    "Reviewed_by_EPC", "Do_not_use"
]

# Crear X con solo genes (drop meta_cols)
X = merged.drop(columns=[c for c in meta_cols if c in merged.columns], errors="ignore")

# quitar columnas completamente vacías
X = X.dropna(axis=1, how="all")

# convertir todo a numérico
X = X.apply(pd.to_numeric, errors="coerce")

# quitar genes totalmente NaN
X = X.dropna(axis=1, how="all")

# rellenar NaN con 0
X = X.fillna(0)

print("Genes finales:", X.shape[1])
print("Muestras:", X.shape[0])
print("Clases (tipos de tumor):", y.nunique())


#imprimir cuantas muestras hay de cada tipo de cancer
print("Distribución de clases (tipos de tumor):")
print(y.value_counts())
X.columns = X.columns.astype(str)
X = X.loc[:, ~X.columns.str.contains(r'\?')]
#verificar que no hay ningun gen con un ? en su nombre
print("Genes finales después de eliminar '?' en nombres:", X.shape[1])
#imprimir los primeros 30 nombres de genes
#print("Primeros 30 nombres de genes:", X.head(30))


#-----------
# Sin PCA
#-----------
# Random Forest sin PCA
# Split train/test (estratificado por clase)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)
# Entrenar Random Forest sobre los datos originales
rf_no_pca = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'   # importante si clases desbalanceadas
)
rf_no_pca.fit(X_train_rf, y_train_rf)
# Predicción y métricas
y_pred_rf_no_pca = rf_no_pca.predict(X_test_rf)
print("=== Classification report (Random Forest sin PCA) ===")
print(classification_report(y_test_rf, y_pred_rf_no_pca, digits=4))
cm_no_pca = confusion_matrix(y_test_rf, y_pred_rf_no_pca)
disp_no_pca = ConfusionMatrixDisplay(confusion_matrix=cm_no_pca)
plt.figure(figsize=(12,10))
disp_no_pca.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - Random Forest sin PCA')
plt.tight_layout()

#  variables mas importantes según Random Forest sin PCA
importances = rf_no_pca.feature_importances_
indices = np.argsort(importances)[::-1]
# Top 30 genes más importantes
top_n = 30
top_genes = X.columns[indices[:top_n]]
top_importances = importances[indices[:top_n]]
plt.figure(figsize=(10,6))
sns.barplot(x=top_importances, y=top_genes, palette='viridis')
plt.title('Top 30 genes más importantes según Random Forest (sin PCA)')
plt.xlabel('Importancia')
plt.ylabel('Genes')
plt.tight_layout()

# randon forest con las 30 variables más importantes
rf_importance = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'
)

rf_importance.fit(X_train_rf[top_genes], y_train_rf)
y_pred_rf_importance = rf_importance.predict(X_test_rf[top_genes])
print("=== Classification report (Random Forest con top 20 genes) ===")
print(classification_report(y_test_rf, y_pred_rf_importance, digits=4))

cm_importance = confusion_matrix(y_test_rf, y_pred_rf_importance)
disp_importance = ConfusionMatrixDisplay(confusion_matrix=cm_importance)
plt.figure(figsize=(12,10))
disp_importance.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - Random Forest con top 20 genes')   
plt.tight_layout()




# naive bayes con las 30 variables más importantes
from sklearn.naive_bayes import GaussianNB  
gnb_importance = GaussianNB()
gnb_importance.fit(X_train_rf[top_genes], y_train_rf)
y_pred_gnb_importance = gnb_importance.predict(X_test_rf[top_genes])
print("=== Classification report (Naive Bayes con top 20 genes) ===")
print(classification_report(y_test_rf, y_pred_gnb_importance, digits=4))
cm_gnb_importance = confusion_matrix(y_test_rf, y_pred_gnb_importance)
disp_gnb_importance = ConfusionMatrixDisplay(confusion_matrix=cm_gnb_importance)
plt.figure(figsize=(12,10))
disp_gnb_importance.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - Naive Bayes con top 20 genes')
plt.tight_layout()

# KNN con las 30 variables más importantes
from sklearn.neighbors import KNeighborsClassifier
knn_importance = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_importance.fit(X_train_rf[top_genes], y_train_rf)
y_pred_knn_importance = knn_importance.predict(X_test_rf[top_genes])
print("=== Classification report (KNN con top 20 genes) ===")
print(classification_report(y_test_rf, y_pred_knn_importance, digits=4))
cm_knn_importance = confusion_matrix(y_test_rf, y_pred_knn_importance)
disp_knn_importance = ConfusionMatrixDisplay(confusion_matrix=cm_knn_importance)
plt.figure(figsize=(12,10))
disp_knn_importance.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - KNN con top 20 genes')
plt.tight_layout()

# regresión logística con las 30 variables más importantes
from sklearn.linear_model import LogisticRegression
logreg_importance = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'
)
logreg_importance.fit(X_train_rf[top_genes], y_train_rf)
y_pred_logreg_importance = logreg_importance.predict(X_test_rf[top_genes])
print("=== Classification report (Logistic Regression con top 20 genes) ===")
print(classification_report(y_test_rf, y_pred_logreg_importance, digits=4))
cm_logreg_importance = confusion_matrix(y_test_rf, y_pred_logreg_importance)
disp_logreg_importance = ConfusionMatrixDisplay(confusion_matrix=cm_logreg_importance)
plt.figure(figsize=(12,10))
disp_logreg_importance.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - Regresión Logística con top 20 genes')
plt.tight_layout()



# redes neuronales con las 30 variables más importantes
from sklearn.neural_network import MLPClassifier    
mlp_importance = MLPClassifier(
    hidden_layer_sizes=(100,50),
    max_iter=500,   
    random_state=RANDOM_STATE
)
mlp_importance.fit(X_train_rf[top_genes], y_train_rf)
y_pred_mlp_importance = mlp_importance.predict(X_test_rf[top_genes])
print("=== Classification report (MLP Classifier con top 20 genes) ===")
print(classification_report(y_test_rf, y_pred_mlp_importance, digits=4))
cm_mlp_importance = confusion_matrix(y_test_rf, y_pred_mlp_importance)
disp_mlp_importance = ConfusionMatrixDisplay(confusion_matrix=cm_mlp_importance)
plt.figure(figsize=(12,10))
disp_mlp_importance.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - MLP Classifier con top 20 genes')
plt.tight_layout()
















#---------------------------
# PCA
#---------------------------


# Codificar etiquetas
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_

# Escalado (muy importante antes de PCA)
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)   # numpy array (n_samples, n_genes)

# PCA: 
pca = PCA(n_components=50, svd_solver='randomized', random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)  # (n_samples, n_components_selected)

print("Número de componentes PCA retenidas:", X_pca.shape[1])
explained_var_ratio = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var_ratio)

# Plot: Scree / varianza explicada acumulada
plt.figure(figsize=(8,5))
plt.plot(np.arange(1, len(explained_var_ratio)+1), cum_var, marker='o')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza explicada acumulada')
plt.title('PCA - Varianza explicada acumulada')
plt.grid(True)
plt.tight_layout()


# PCA: 35 componentes
pca = PCA(n_components=35, svd_solver='randomized', random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)  # (n_samples, n_components_selected)

print("Número de componentes PCA retenidas:", X_pca.shape[1])

# Split train/test (estratificado por clase)
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_enc, test_size=0.3, stratify=y_enc, random_state=RANDOM_STATE
)

# grafica de las dos componentes principales más importantes
plt.figure(figsize=(8,8))   
pc1_idx, pc2_idx = 0, 1  # primeras dos componentes
sns.scatterplot(
    x=X_pca[:, pc1_idx], y=X_pca[:, pc2_idx],
    hue=y, palette='tab20', alpha=0.7
)
plt.xlabel(f'PC{pc1_idx+1} ({explained_var_ratio[pc1_idx]*100:.2f}% varianza)')
plt.ylabel(f'PC{pc2_idx+1} ({explained_var_ratio[pc2_idx]*100:.2f}% varianza)')
plt.title('Proyección PCA de las muestras ')
plt.legend(title='Tipo de tumor', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


#---------------------------
# Random Forest sobre PCA
#---------------------------


# Entrenar Random Forest sobre las componentes PCA
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'   # importante si clases desbalanceadas
)
rf.fit(X_train, y_train)

# Predicción y métricas
y_pred = rf.predict(X_test)
print("=== Classification report ===")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

# Matriz de confusión (visual)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(12,10))
disp.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - Random Forest sobre PCA')
plt.tight_layout()



#---------------------------
# Clasificación con Naive Bayes sobre PCA
#---------------------------

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print("=== Classification report (Naive Bayes) ===")
print(classification_report(y_test, y_pred_gnb, target_names=class_names, digits=4))
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
disp_gnb = ConfusionMatrixDisplay(confusion_matrix=cm_gnb, display_labels=class_names)
plt.figure(figsize=(12,10))
disp_gnb.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - Naive Bayes sobre PCA')
plt.tight_layout()


#---------------------------
# Clasificación con KNN sobre PCA
#---------------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("=== Classification report (KNN) ===")
print(classification_report(y_test, y_pred_knn, target_names=class_names, digits=4))
cm_knn = confusion_matrix(y_test, y_pred_knn)   
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=class_names)
plt.figure(figsize=(12,10))
disp_knn.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - KNN sobre PCA')
plt.tight_layout()

#---------------------------
# Clasificación con regresión logística sobre PCA
#---------------------------    
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'
)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("=== Classification report (Logistic Regression) ===")
print(classification_report(y_test, y_pred_logreg, target_names=class_names, digits=4))
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
disp_logreg = ConfusionMatrixDisplay(confusion_matrix=cm_logreg, display_labels=class_names)
plt.figure(figsize=(12,10))
disp_logreg.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - Regresión Logística sobre PCA')
plt.tight_layout()

#---------------------------
# Clasificación con redes neuronales sobre PCA
#---------------------------
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100,50),
    max_iter=500,
    random_state=RANDOM_STATE
)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("=== Classification report (MLP Classifier) ===")
print(classification_report(y_test, y_pred_mlp, target_names=class_names, digits=4))
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=class_names)
plt.figure(figsize=(12,10))
disp_mlp.plot(include_values=True, xticks_rotation='vertical', values_format='d', ax=plt.gca())
plt.title('Matriz de confusión - MLP Classifier sobre PCA')
plt.tight_layout()