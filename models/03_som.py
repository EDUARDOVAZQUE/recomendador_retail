import json
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os
import random
from collections import defaultdict, Counter

def main():
    config_path = '../config/som.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Iniciando entrenamiento de SOM (Basket-based)...")

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

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(X_raw)
    y = np.array(y_raw)
    
    grid_size = config.get('grid_size', [10, 10])
    som = MiniSom(grid_size[0], grid_size[1], X.shape[1], 
                  sigma=config.get('sigma', 1.0), 
                  learning_rate=config.get('learning_rate', 0.5), 
                  random_seed=42)
                  
    print("Entrenando SOM... (puede tardar un poco)")
    som.train_random(X, config.get('num_iteration', 1000))
    
    print("Mapeando nodos a recomendaciones...")
    node_recommendations = defaultdict(list)
    for i, x in enumerate(X):
        winner = som.winner(x)
        node_recommendations[winner].append(y[i])
        
    # Guardar el ranking de productos por nodo
    node_ranking = {}
    for node, targets in node_recommendations.items():
        # Cuenta frecuencias y ordena de mayor a menor
        most_common = [item for item, count in Counter(targets).most_common()]
        node_ranking[node] = most_common

    # Si hay nodos que nunca ganaron, les asignamos los items más populares en general
    global_most_common = [item for item, count in Counter(y).most_common()]
    
    export_data = {
        'model': som,
        'binarizer': mlb,
        'prod_cat_map': prod_cat_map,
        'all_products': list(mlb.classes_),
        'node_ranking': node_ranking,
        'global_ranking': global_most_common
    }

    export_path = '../export/som_basket.pkl'
    os.makedirs('../export', exist_ok=True)
    joblib.dump(export_data, export_path)
    print(f"Modelo exportado exitosamente a: {export_path}")

if __name__ == "__main__":
    main()
