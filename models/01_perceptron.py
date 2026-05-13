import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os
import random

def main():
    config_path = '../config/perceptron.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Iniciando entrenamiento del Perceptrón (Basket-based)...")

    data_path = '../sales.csv'
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Crear mapeo de producto a categoría para usar en la API
    prod_cat_map = df[['product', 'category']].drop_duplicates().set_index('product')['category'].to_dict()
    
    # Agrupar compras por carrito (invoice_id)
    baskets = df.groupby('invoice_id')['product'].apply(list).reset_index()
    
    # Solo usamos carritos con más de 1 producto para poder predecir 1 extra
    baskets = baskets[baskets['product'].apply(len) > 1]
    
    X_raw = []
    y_raw = []
    
    # Construir el dataset: escogemos 1 producto aleatorio como objetivo (y)
    # y usamos el resto del carrito como entrada (X)
    random.seed(42) # para reproducibilidad
    for items in baskets['product']:
        target = random.choice(items)
        # El carrito de entrada son los items menos la primera ocurrencia del target
        input_basket = items.copy()
        input_basket.remove(target)
        
        if len(input_basket) == 0:
            continue
        X_raw.append(input_basket)
        y_raw.append(target)
        
    print(f"Dataset generado con {len(X_raw)} ejemplos de carritos.")

    # Convertir los carritos a vectores binarios (ej. [1, 0, 1, 0...])
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(X_raw)
    y = np.array(y_raw)

    # Inicializar perceptrón
    model = Perceptron(
        eta0=config.get('learning_rate', 0.01),
        max_iter=config.get('epochs', 1000),
        random_state=42,
        class_weight='balanced'
    )

    print("Entrenando modelo perceptrón multicapa... (esto tomará un momento)")
    model.fit(X, y)
    
    score = model.score(X, y)
    print(f"Precisión (Accuracy) en entrenamiento: {score:.4f}")

    # Empaquetamos todo lo necesario para la API
    export_data = {
        'model': model,
        'binarizer': mlb,
        'prod_cat_map': prod_cat_map,
        'all_products': list(mlb.classes_)
    }

    export_path = '../export/perceptron_basket.pkl'
    os.makedirs('../export', exist_ok=True)
    joblib.dump(export_data, export_path)
    print(f"Modelo exportado exitosamente a: {export_path}")

if __name__ == "__main__":
    main()
