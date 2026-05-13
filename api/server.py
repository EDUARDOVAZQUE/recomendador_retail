from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI(title="Recomendador Retail API")

# Definición de la arquitectura del Transformer para poder cargar los pesos
class MicroTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super(MicroTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        src_key_padding_mask = (x == 0)
        embedded = self.embedding(x)
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        mask = (~src_key_padding_mask).unsqueeze(-1).float()
        sum_embeddings = (encoded * mask).sum(dim=1)
        valid_lengths = mask.sum(dim=1).clamp(min=1)
        mean_pooled = sum_embeddings / valid_lengths
        logits = self.fc_out(mean_pooled)
        return logits

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerProfile(BaseModel):
    cart: List[str]
    model_choice: str

# Diccionario para cargar los modelos en memoria
models = {}

def load_models():
    perceptron_path = "../export/perceptron_basket.pkl"
    if os.path.exists(perceptron_path):
        models["perceptron"] = joblib.load(perceptron_path)
        print("✅ Modelo Perceptrón (Basket) cargado correctamente en memoria.")
    
    mlp_path = "../export/mlp_basket.pkl"
    if os.path.exists(mlp_path):
        models["mlp"] = joblib.load(mlp_path)
        print("✅ Modelo MLP (Basket) cargado correctamente en memoria.")
        
    som_path = "../export/som_basket.pkl"
    if os.path.exists(som_path):
        models["som"] = joblib.load(som_path)
        print("✅ Modelo SOM (Basket) cargado correctamente en memoria.")
        
    transformer_path = "../export/transformer_basket.pt"
    transformer_meta = "../export/transformer_meta.pkl"
    if os.path.exists(transformer_path) and os.path.exists(transformer_meta):
        meta = joblib.load(transformer_meta)
        model_config = meta['model_config']
        model = MicroTransformer(**model_config)
        model.load_state_dict(torch.load(transformer_path, map_location=torch.device('cpu')))
        model.eval()
        models["transformer"] = {
            "model": model,
            "meta": meta,
            "all_products": meta['all_products']
        }
        print("✅ Modelo Micro-Transformer cargado correctamente en memoria.")

# Usar el evento de inicio correcto
@app.on_event("startup")
def startup_event():
    load_models()

@app.get("/products")
def get_products():
    """Devuelve la lista de productos conocidos por el modelo"""
    if "perceptron" in models:
        return {"products": models["perceptron"]["all_products"]}
    elif "mlp" in models:
        return {"products": models["mlp"]["all_products"]}
    elif "som" in models:
        return {"products": models["som"]["all_products"]}
    elif "transformer" in models:
        return {"products": models["transformer"]["all_products"]}
    return {"products": []}

@app.get("/api/home")
def api_home():
    try:
        df = pd.read_csv("../sales.csv")
        stores = df['city'].unique().tolist()
        
        product_sales = df.groupby('product')['quantity'].sum().reset_index()
        top_products = product_sales.sort_values(by='quantity', ascending=False).head(5)
        top_products_list = [{"product": row['product'], "total_qty": int(row['quantity'])} for _, row in top_products.iterrows()]
            
        # Ofertas (aleatorias estables)
        np.random.seed(42)
        offers_df = df[['city', 'product', 'unit_price']].drop_duplicates().sample(min(3, len(df)), random_state=42)
        offers_list = [{"branch": row['city'], "product": row['product'], "price": float(row['unit_price']) * 0.8} for _, row in offers_df.iterrows()]
            
        return {"stores": stores, "top_products": top_products_list, "offers": offers_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecast")
def api_forecast():
    try:
        df = pd.read_csv("../sales.csv")
        
        # Procesar fechas con formato mixto para evitar el warning
        df['date_obj'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)
        
        # Nos aseguramos de tener datos agrupables por fecha
        daily_invoices = df.groupby('date_obj')['invoice_id'].nunique().reset_index()
        daily_invoices = daily_invoices.set_index('date_obj').sort_index()
        
        def generate_prediction(history_counts, last_date, freq, periods):
            last_avg = np.mean(history_counts[-min(3, len(history_counts)):]) if len(history_counts) > 0 else 100
            pred_dates = []
            pred_counts = []
            
            import datetime
            import calendar
            
            np.random.seed(42)
            current_date = last_date
            
            for i in range(1, periods + 1):
                if freq == 'D':
                    current_date += datetime.timedelta(days=1)
                    noise = np.random.normal(0, last_avg * 0.1)
                    multiplier = 1.4 if current_date.weekday() >= 5 else 0.9
                elif freq == 'W':
                    current_date += datetime.timedelta(weeks=1)
                    noise = np.random.normal(0, last_avg * 0.1)
                    multiplier = 1.0 # Semanas estables
                elif freq == 'M':
                    days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]
                    current_date += datetime.timedelta(days=days_in_month)
                    noise = np.random.normal(0, last_avg * 0.05)
                    multiplier = 1.5 if current_date.month == 12 else 1.0
                elif freq == 'Y':
                    current_date = current_date.replace(year=current_date.year + 1)
                    noise = np.random.normal(0, last_avg * 0.05)
                    multiplier = 1.05 # Crecimiento anual 5%
                
                pred_val = int(max(0, last_avg * multiplier + noise))
                
                if freq == 'D': fmt = '%Y-%m-%d'
                elif freq == 'W': fmt = '%Y-W%W'
                elif freq == 'M': fmt = '%Y-%m'
                else: fmt = '%Y'
                
                pred_dates.append(current_date.strftime(fmt))
                pred_counts.append(pred_val)
                
            return pred_dates, pred_counts

        def get_resolution_data(resampled_df, freq_enum, pred_periods):
            if resampled_df.empty:
                return {"history": {"dates": [], "counts": []}, "prediction": {"dates": [], "counts": []}}
                
            counts = resampled_df['invoice_id'].tolist()
            
            if freq_enum == 'D':
                dates = resampled_df.index.strftime('%Y-%m-%d').tolist()
            elif freq_enum == 'W':
                dates = resampled_df.index.strftime('%Y-W%W').tolist()
            elif freq_enum == 'M':
                dates = resampled_df.index.strftime('%Y-%m').tolist()
            else:
                dates = resampled_df.index.strftime('%Y').tolist()
                
            # Limitar a un número manejable de puntos en historia
            limit = 30 if freq_enum == 'D' else (24 if freq_enum == 'M' else len(dates))
            hist_dates = dates[-limit:]
            hist_counts = counts[-limit:]
            
            last_date = resampled_df.index[-1]
            pred_dates, pred_counts = generate_prediction(counts, last_date, freq_enum, pred_periods)
            
            return {
                "history": {
                    "dates": hist_dates,
                    "counts": hist_counts
                },
                "prediction": {
                    "dates": pred_dates,
                    "counts": pred_counts
                }
            }

        # Generar las agregaciones
        res_D = daily_invoices.resample('D').sum()
        res_W = daily_invoices.resample('W-MON').sum()
        res_M = daily_invoices.resample('ME').sum()
        res_Y = daily_invoices.resample('YE').sum()

        response = {
            "day": get_resolution_data(res_D, 'D', 7),
            "week": get_resolution_data(res_W, 'W', 4),
            "month": get_resolution_data(res_M, 'M', 3),
            "year": get_resolution_data(res_Y, 'Y', 1)
        }
        
        # Heatmap del mes actual
        if not res_D.empty:
            last_month = res_D.index[-1].month
            last_year = res_D.index[-1].year
            current_month_data = res_D[(res_D.index.year == last_year) & (res_D.index.month == last_month)]
            
            heatmap_data = []
            for date, row in current_month_data.iterrows():
                heatmap_data.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "day": date.day,
                    "weekday": date.weekday(), # 0=Monday
                    "count": int(row['invoice_id'])
                })
            response["heatmap"] = {
                "month": last_month,
                "year": last_year,
                "data": heatmap_data
            }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(profile: CustomerProfile):
    model_key = profile.model_choice.lower()
    
    if model_key not in models:
        raise HTTPException(status_code=400, detail=f"Modelo '{model_key}' no entrenado o no encontrado.")
    
    model_data = models[model_key]
    model = model_data['model']
    
    if len(profile.cart) == 0:
        raise HTTPException(status_code=400, detail="El carrito está vacío.")
        
    if model_key == "transformer":
        meta = model_data["meta"]
        prod_cat_map = meta["prod_cat_map"]
    else:
        binarizer = model_data['binarizer']
        prod_cat_map = model_data['prod_cat_map']
        
        # Vectorizar el carrito (convertir la lista de strings a matriz binaria [1, 0, 1...])
        try:
            # transform espera una lista de listas
            X_input = binarizer.transform([profile.cart])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error procesando productos: {str(e)}")
        
    try:
        if model_key == "som":
            # Para SOM usamos el nodo ganador y su ranking asociado
            winner = model.winner(X_input[0])
            node_ranking = model_data['node_ranking']
            global_ranking = model_data['global_ranking']
            
            candidates = node_ranking.get(winner, global_ranking)
            
            recommended_product = None
            for candidate in candidates:
                if candidate not in profile.cart:
                    recommended_product = candidate
                    break
                    
            if not recommended_product:
                for candidate in global_ranking:
                    if candidate not in profile.cart:
                        recommended_product = candidate
                        break
        elif model_key == "transformer":
            meta = model_data["meta"]
            prod_to_idx = meta["prod_to_idx"]
            idx_to_prod = meta["idx_to_prod"]
            
            # Convertir carrito a índices
            input_indices = [prod_to_idx[p] for p in profile.cart if p in prod_to_idx]
            if not input_indices:
                raise HTTPException(status_code=400, detail="Productos desconocidos para el Transformer.")
                
            # Rellenar a max_len 5
            max_len = 5
            input_indices = input_indices[:max_len]
            padding_length = max_len - len(input_indices)
            padded_input = input_indices + [0] * padding_length
            
            x_tensor = torch.tensor([padded_input], dtype=torch.long)
            
            with torch.no_grad():
                logits = model(x_tensor)
                scores = torch.softmax(logits, dim=1)[0].numpy()
                
            sorted_indices = np.argsort(scores)[::-1]
            
            recommended_product = None
            for idx in sorted_indices:
                # idx es 0-based, idx_to_prod es 1-based (del entrenamiento)
                candidate = idx_to_prod.get(idx + 1)
                if candidate and candidate not in profile.cart:
                    recommended_product = candidate
                    break
                    
            if not recommended_product:
                recommended_product = idx_to_prod.get(sorted_indices[0] + 1)
                
        else:
            # Usar ranking (probabilidades o decisión) para evitar productos ya en el carrito
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_input)[0]
            else:
                scores = model.decision_function(X_input)[0]
                
            # Ordenar de mayor a menor score
            sorted_indices = np.argsort(scores)[::-1]
            
            recommended_product = None
            for idx in sorted_indices:
                candidate = model.classes_[idx]
                if candidate not in profile.cart:
                    recommended_product = candidate
                    break
                    
            if not recommended_product:
                # Fallback de seguridad
                recommended_product = model.predict(X_input)[0]
            
        category = prod_cat_map.get(recommended_product, "Desconocida")
        
        return {
            "status": "success", 
            "recommended_product": recommended_product,
            "recommended_category": category,
            "model_used": profile.model_choice
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")

if __name__ == "__main__":
    print("Iniciando servidor API en http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
