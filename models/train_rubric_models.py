import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import os

# Create export folder if not exists
os.makedirs('../export', exist_ok=True)

print("Cargando dataset...")
# Load dataset
df = pd.read_csv('../sales.csv')

# -------------------------------------------------------------
# 1. MODELO DE REGRESIÓN: Predecir la afluencia/tráfico (cantidad total de productos por hora y día)
# Objetivo: Predecir 'quantity' basándonos en la hora y otros factores.
# -------------------------------------------------------------
print("\n--- ENTRENANDO MODELO DE REGRESIÓN ---")
df['date_obj'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)
df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour
df['day_of_week'] = df['date_obj'].dt.dayofweek

# Agruparemos para obtener tráfico por hora y día
reg_df = df.groupby(['date_obj', 'hour', 'day_of_week'])['quantity'].sum().reset_index()

X_reg = reg_df[['hour', 'day_of_week']]
y_reg = reg_df['quantity']

# Hiperparámetros justificados: RandomForest porque captura no linealidades en tráfico.
# n_estimators=100 y max_depth=10 para evitar sobreajuste en datos diarios limitados.
reg_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Validación cruzada K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
mae_scores = []
r2_scores = []

for train_idx, test_idx in kf.split(X_reg):
    X_tr, X_te = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
    y_tr, y_te = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    
    reg_model.fit(X_tr, y_tr)
    preds = reg_model.predict(X_te)
    
    rmse_scores.append(np.sqrt(mean_squared_error(y_te, preds)))
    mae_scores.append(mean_absolute_error(y_te, preds))
    r2_scores.append(r2_score(y_te, preds))

print(f"Resultados de Regresión (5-Fold CV):")
print(f"RMSE Promedio: {np.mean(rmse_scores):.4f}")
print(f"MAE Promedio:  {np.mean(mae_scores):.4f}")
print(f"R² Promedio:   {np.mean(r2_scores):.4f}")

# Entrenar modelo final y exportar
reg_model.fit(X_reg, y_reg)
joblib.dump(reg_model, '../export/traffic_regressor.pkl')
print("Modelo de regresión exportado a '../export/traffic_regressor.pkl'")

# -------------------------------------------------------------
# 2. MODELO DE CLASIFICACIÓN: Predecir Categoría (Ontología / Recomendación post-compra)
# Objetivo: Predecir la categoría basada en el género y tipo de cliente
# -------------------------------------------------------------
print("\n--- ENTRENANDO MODELO DE CLASIFICACIÓN ---")
# Codificación categórica simple
df_clf = df.copy()
df_clf = df_clf.dropna(subset=['gender', 'customer_type', 'category'])
df_clf['gender_code'] = df_clf['gender'].astype('category').cat.codes
df_clf['customer_code'] = df_clf['customer_type'].astype('category').cat.codes

X_clf = df_clf[['gender_code', 'customer_code', 'hour']]
y_clf = df_clf['category']

clf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

acc_scores = cross_val_score(clf_model, X_clf, y_clf, cv=5, scoring='accuracy')
print(f"Resultados de Clasificación (5-Fold CV):")
print(f"Accuracy Promedio: {np.mean(acc_scores):.4f}")

clf_model.fit(X_clf, y_clf)

# Guardar diccionarios para revertir códigos
categories = df['category'].astype('category').cat.categories.tolist()
genders = df['gender'].astype('category').cat.categories.tolist()
customers = df['customer_type'].astype('category').cat.categories.tolist()

meta_clf = {
    'model': clf_model,
    'categories': categories,
    'genders': genders,
    'customers': customers
}
joblib.dump(meta_clf, '../export/category_classifier.pkl')
print("Modelo de clasificación exportado a '../export/category_classifier.pkl'")

print("\n--- FINALIZADO ---")
