import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import random

# Definición de la arquitectura del Transformer
class MicroTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super(MicroTransformer, self).__init__()
        # Embedding: +1 por el padding (índice 0)
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        
        # Capa del Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Capa de salida (predice probabilidades para todo el vocabulario)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        # padding mask: True donde hay padding (índice 0)
        src_key_padding_mask = (x == 0)
        
        embedded = self.embedding(x) # (batch_size, seq_len, d_model)
        
        # Pasar por Transformer
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        
        # Hacer pooling (promedio de los embeddings que NO son padding)
        # Crear máscara para ignorar padding en el promedio
        mask = (~src_key_padding_mask).unsqueeze(-1).float()
        sum_embeddings = (encoded * mask).sum(dim=1)
        valid_lengths = mask.sum(dim=1).clamp(min=1)
        mean_pooled = sum_embeddings / valid_lengths
        
        # Predicción final
        logits = self.fc_out(mean_pooled)
        return logits

def main():
    config_path = '../config/transformer.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    print("Iniciando entrenamiento del Micro-Transformer con PyTorch...")
    
    data_path = '../sales.csv'
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo {data_path}")
        return
        
    df = pd.read_csv(data_path)
    prod_cat_map = df[['product', 'category']].drop_duplicates().set_index('product')['category'].to_dict()
    
    # Crear vocabulario
    all_products = sorted(df['product'].unique())
    vocab_size = len(all_products)
    # producto a índice (1-based)
    prod_to_idx = {p: i+1 for i, p in enumerate(all_products)}
    # índice a producto
    idx_to_prod = {i+1: p for i, p in enumerate(all_products)}
    
    baskets = df.groupby('invoice_id')['product'].apply(list).reset_index()
    baskets = baskets[baskets['product'].apply(len) > 1]
    
    X_raw = []
    y_raw = []
    
    random.seed(42)
    max_len = 5 # Asumimos máximo 5 elementos en la canasta como entrada
    
    for items in baskets['product']:
        target = random.choice(items)
        input_basket = items.copy()
        input_basket.remove(target)
        
        if len(input_basket) == 0:
            continue
            
        # Limitar a max_len y rellenar con 0s (padding) si es menor
        input_indices = [prod_to_idx[p] for p in input_basket[:max_len]]
        padding_length = max_len - len(input_indices)
        padded_input = input_indices + [0] * padding_length
        
        # Target index es 0-based para CrossEntropyLoss (0 a vocab_size-1)
        target_idx = prod_to_idx[target] - 1
        
        X_raw.append(padded_input)
        y_raw.append(target_idx)
        
    print(f"Dataset generado con {len(X_raw)} ejemplos.")
    
    # Convertir a Tensores
    X_tensor = torch.tensor(X_raw, dtype=torch.long)
    y_tensor = torch.tensor(y_raw, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    
    # Inicializar Modelo
    model = MicroTransformer(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 32),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 2),
        dim_feedforward=config.get('dim_feedforward', 64)
    )
    
    # Pérdida y Optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    epochs = config.get('epochs', 10)
    print(f"Comenzando ciclo de entrenamiento ({epochs} épocas)...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calcular precisión
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Época [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        
    # Exportar
    os.makedirs('../export', exist_ok=True)
    
    # Guardar pesos (State Dict)
    torch.save(model.state_dict(), '../export/transformer_basket.pt')
    
    # Guardar metadatos
    meta_data = {
        'prod_to_idx': prod_to_idx,
        'idx_to_prod': idx_to_prod,
        'prod_cat_map': prod_cat_map,
        'all_products': all_products,
        'model_config': {
            'vocab_size': vocab_size,
            'd_model': config.get('d_model', 32),
            'nhead': config.get('nhead', 4),
            'num_layers': config.get('num_layers', 2),
            'dim_feedforward': config.get('dim_feedforward', 64)
        }
    }
    joblib.dump(meta_data, '../export/transformer_meta.pkl')
    
    print("Modelo Transformer exportado exitosamente a /export/")

if __name__ == "__main__":
    main()
