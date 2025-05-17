#!/usr/bin/env python
# coding: utf-8

# ### IMPORTACIONES

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf


# ### CARGAR MODELO ENTRENADO Y SCALER

# In[2]:


MODEL_PATH = 'trained_model.keras'
SCALER_PATH = 'scaler.pkl'


# In[3]:


model = tf.keras.models.load_model(MODEL_PATH)


# In[4]:


with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)


# ### FAST-API SETUP

# In[5]:


app = FastAPI(title="S&P500 Predictor API")


# In[6]:


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Solo permite Angular local
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todas las cabeceras (headers)
)  


# ### MODELOS PYDANTIC

# In[7]:


class DateRangePast(BaseModel):
    start_date: str
    end_date: str


# In[8]:


class DateRangeFuture(BaseModel):
    days: int


# ### END-POINT 1 -> ALL PRICES HISTORY

# In[9]:


historical_data = None 

@app.get("/history")
def get_history():
    global historical_data

    if historical_data is None:
        try:
            # Descargar histórico completo
            df = yf.download('^GSPC', period='max', auto_adjust=True)

            if df.empty:
                raise HTTPException(status_code=404, detail="No se encontraron datos históricos del S&P 500.")

            # Acceder correctamente a la columna 'Close' con MultiIndex
            close_series = df[('Close', '^GSPC')]

            # Fechas y valores
            dates = close_series.index.strftime('%Y-%m-%d').tolist()
            values = close_series.round(2).tolist()

            # Guardar en caché
            historical_data = {
                "dates": dates,
                "values": values,
                "total_points": len(values),
                "message": "Histórico completo del S&P 500 desde inicio hasta hoy."
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al obtener datos históricos: {str(e)}")

    return historical_data


# ### END-POINT 2 -> NORMAL PREDICTION FOR THE PAST

# In[10]:


last_prediction_metrics = None


# In[11]:


@app.post("/normal_prediction")
def normal_prediction(date_range_past: DateRangePast):
    global last_prediction_metrics  # Para actualizar métricas globales

    try:
        # Calcular fecha extendida (pedimos más datos para generar las primeras ventanas)
        start_date_obj = datetime.strptime(date_range_past.start_date, "%Y-%m-%d")
        extended_start_date = start_date_obj - timedelta(days=90)  # Mejor más de 60 para asegurarse
        extended_start_date_str = extended_start_date.strftime("%Y-%m-%d")

        # Descargar datos desde fecha extendida
        df = yf.download('^GSPC', start=extended_start_date_str, end=date_range_past.end_date, auto_adjust=True)

        if df.empty:
            raise HTTPException(status_code=404, detail="No se encontraron datos para las fechas especificadas.")

        df_closing_prices = df[['Close']]  # Mantener solo la columna 'Close'

        # Usar scaler original (entrenado)
        data = scaler.transform(df_closing_prices)

        # Generar secuencias
        X_test, y_real_scaled = [], []
        for i in range(60, len(data)):
            X_test.append(data[i-60:i, 0])      # Ventana de 60 días
            y_real_scaled.append(data[i, 0])    # Valor real para esa secuencia

        if not X_test:
            raise HTTPException(status_code=400, detail="No hay suficientes datos (mínimo 60 días).")

        X_test = np.array(X_test).reshape((len(X_test), 60, 1))

        # Hacer predicciones
        y_pred_scaled = model.predict(X_test)

        # Desescalar predicciones
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_real = scaler.inverse_transform(np.array(y_real_scaled).reshape(-1, 1))

        # Fechas alineadas con las predicciones (desde la posición 60)
        all_dates = df_closing_prices.index[60:].strftime('%Y-%m-%d').tolist()

        # Filtrar las fechas y predicciones a partir de la fecha real solicitada (start_date)
        start_index = next((i for i, date in enumerate(all_dates) if date >= date_range_past.start_date), None)
        final_dates = all_dates[start_index:]
        final_predictions = y_pred.flatten().tolist()[start_index:]
        final_real_values = y_real[start_index:].flatten().tolist()  # Valores reales alineados

        # Filtrar también y_real para calcular métricas solo desde start_date
        y_real_filtered = y_real[start_index:]
        y_pred_filtered = np.array(final_predictions).reshape(-1, 1)  # Asegurarse que esté en formato correcto para métricas

        # Calcular métricas desde start_date
        rmse = math.sqrt(mean_squared_error(y_real_filtered, y_pred_filtered))
        mae = mean_absolute_error(y_real_filtered, y_pred_filtered)
        relative_rmse = (rmse / np.mean(y_real_filtered)) * 100

        # Guardar métricas para endpoint /metrics
        last_prediction_metrics = {
            "Num_samples": len(X_test),
            "Start_date": date_range_past.start_date,
            "End_date": date_range_past.end_date,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "Relative_RMSE_%": round(relative_rmse, 2)
        }

        return {
            "predictions": final_predictions,
            "real_values": final_real_values,
            "dates": final_dates,
            "message": f"Predicciones de S&P500 desde {date_range_past.start_date} hasta {date_range_past.end_date}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ### END-POINT 3 -> RECURSIVE PREDICTION FOR THE FUTURE

# In[12]:


def create_dataset(data, time_step):
    X_total, y_total = [], []
    
    for i in range(time_step, len(data)):
        X_total.append(data[i-time_step:i, 0])  # Los últimos 60 días
        y_total.append(data[i, 0])  # El precio de cierre del día siguiente
    
    return np.array(X_total), np.array(y_total)


# In[13]:


df = yf.download('^GSPC', period='max', auto_adjust=True)
df_closing_prices = df[['Close']]


# In[14]:


training_set_scaled = scaler.fit_transform(df_closing_prices)

last_days = training_set_scaled[-61:].astype(np.float32)

X_test, y_test = create_dataset(last_days, 60)


# In[15]:


start_date = datetime.today()


# In[16]:


def generate_future_dates(num_days: int):
    future_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days + 1)]
    return future_dates


# In[17]:


@app.post("/recursive_prediction")
def recursive_prediction(date_range_future: DateRangeFuture):
    global last_prediction_metrics  # Para actualizar métricas globales

    try:
        # Copia de los últimos 60 valores de X_test
        X_test_selection = training_set_scaled[-60:].flatten()
        
        # Lista para almacenar predicciones futuras
        future_prediction = []
        
        # Diferencia entre el último valor real y el anterior (para ajustar tendencia correctamente)
        real_trend = training_set_scaled[-1] - training_set_scaled[-2]
        
        for _ in range(date_range_future.days):  # 20 días financieros ~ 1 mes normal
            # Convertir la secuencia a tensor de TensorFlow correctamente
            sequence_for_pred_tensor = tf.convert_to_tensor(X_test_selection.reshape(1, 60, 1), dtype=tf.float32)
        
            # Predecir el siguiente valor
            next_day_prediction = model(sequence_for_pred_tensor, training=False).numpy().flatten()[0]
        
            # Comparar la última predicción con el último valor real
            last_real_value = training_set_scaled[-1]
            last_predicted_value = future_prediction[-1] if future_prediction else last_real_value
        
            # Ajuste de tendencia con comparación real
            trend_adjustment = (last_real_value - last_predicted_value) * 0.5  # Corrige la desviación en cada iteración
            next_day_prediction += np.random.normal(0, 0.005) + trend_adjustment
        
            # Guardar la predicción
            future_prediction.append(next_day_prediction)
        
            # Actualizar la secuencia para la siguiente iteración
            X_test_selection = np.append(X_test_selection[1:], next_day_prediction)

        # Transformar predicciones a escala original
        prediction_original_scale = scaler.inverse_transform(np.array(future_prediction).reshape(-1, 1))
        
        # Preparar fechas
        end_date = start_date + timedelta(days=date_range_future.days)

        return {
            "predictions": prediction_original_scale.flatten().tolist(),
            "dates": generate_future_dates(date_range_future.days),
            "message": f"Predicciones de S&P500 desde {start_date.strftime('%Y-%m-%d')} hasta {end_date.strftime('%Y-%m-%d')}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ### END-POINT 4 -> ERROR METRICS

# In[18]:


@app.post("/metrics")
def get_metrics():
    if last_prediction_metrics is None:
        raise HTTPException(status_code=404, detail="No se ha realizado ninguna predicción aún.")
    return last_prediction_metrics

