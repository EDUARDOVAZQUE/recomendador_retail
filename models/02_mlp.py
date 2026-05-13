import json
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os
import random

def main():
    config_path = '../config/mlp.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Iniciando entrenamiento de MLP (Basket-based)...")

    data_path = '../sales.csv'
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    prod_cat_map = df[['product', 'category']].drop_duplicates().set_index('product')['category'].to_dict()
    
    baskets = df.groupby('invoice_id')['product'].apply(list).reset_index()
    baskets = baskets[baskets['product'].apply(len) > 1]
    
    X_raw = []
    y_raw = []
    
    random.seed(42)
    for items in baskets['product']:
        target = random.choice(items)
        input_basket = items.copy()
        input_basket.remove(target)
        
        if len(input_basket) == 0:
            continue
        X_raw.append(input_basket)
        y_raw.append(target)
        
    print(f"Dataset generado con {len(X_raw)} ejemplos.")

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(X_raw)
    y = np.array(y_raw)

    hidden_layers = tuple(config.get('hidden_layers', [16, 8]))
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=config.get('activation', 'relu'),
        solver=config.get('solver', 'adam'),
        max_iter=config.get('max_iter', 500),
        random_state=42
    )

    print("Entrenando MLP... (esto puede demorar unos segundos)")
    model.fit(X, y)
    
    score = model.score(X, y)
    print(f"Precisión (Accuracy) en entrenamiento: {score:.4f}")

    export_data = {
        'model': model,
        'binarizer': mlb,
        'prod_cat_map': prod_cat_map,
        'all_products': list(mlb.classes_)
    }

    export_path = '../export/mlp_basket.pkl'
    os.makedirs('../export', exist_ok=True)
    joblib.dump(export_data, export_path)
    print(f"Modelo exportado exitosamente a: {export_path}")

if __name__ == "__main__":
    main()
