import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.interpolate import interp1d

class WeightedMeanSquaredError(tf.keras.losses.Loss):
    """Loss customizada que considera pesos dos dados"""
    def __init__(self, name="weighted_mse"):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred, sample_weight=None):
        mse = tf.keras.losses.MeanSquaredError(y_true, y_pred)
        if sample_weight is not None:
            mse = tf.multiply(mse, sample_weight)
        return tf.reduce_mean(mse)

def preprocess_data(data, method='robust'):
    """
    Pré-processamento avançado dos dados
    """
    # Remove outliers usando IQR
    data_clean = data.copy()
    Q1 = np.nanpercentile(data_clean, 25)
    Q3 = np.nanpercentile(data_clean, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Substitui outliers por NaN temporariamente
    outlier_mask = (data_clean < lower_bound) | (data_clean > upper_bound)
    data_clean[outlier_mask] = np.nan
    
    # Escolhe o scaler baseado no método
    if method == 'robust':
        scaler = RobustScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Aplica suavização se necessário
    if len(data_clean[~np.isnan(data_clean)]) > 10:
        # Suavização usando filtro Savitzky-Golay
        valid_data = data_clean[~np.isnan(data_clean)]
        if len(valid_data) > 5:
            try:
                window_length = min(11, len(valid_data) if len(valid_data) % 2 == 1 else len(valid_data) - 1)
                if window_length >= 3:
                    smoothed = signal.savgol_filter(valid_data, window_length, 2)
                    data_clean[~np.isnan(data_clean)] = smoothed
            except:
                pass  # Se falhar, continua com dados originais
    
    return data_clean, scaler

def create_weighted_dataset(X, weights, look_back=3):
    Xs, ys, sample_weights = [], [], []
    
    for i in range(len(X) - look_back):
        v = X[i:i + look_back]
        Xs.append(v)
        ys.append(X[i + look_back])
        window_weight = np.mean(weights[i:i + look_back]) * weights[i + look_back]
        sample_weights.append(window_weight)
    
    # Adicione uma dimensão de features (1 feature)
    Xs = np.array(Xs)
    Xs = np.expand_dims(Xs, axis=-1)  # Transforma em [amostras, timesteps, 1]
    
    return Xs, np.array(ys), np.array(sample_weights)

def create_advanced_lstm(units, train_shape, learning_rate, dropout_rate=0.2):
    """Cria modelo LSTM mais robusto"""
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=[train_shape[1], train_shape[2]]),
        Dropout(dropout_rate),
        BatchNormalization(),
        
        LSTM(units=units//2, return_sequences=True),
        Dropout(dropout_rate),
        BatchNormalization(),
        
        LSTM(units=units//4),
        Dropout(dropout_rate),
        
        Dense(units//8, activation='relu'),
        Dropout(dropout_rate/2),
        Dense(1)
    ])
    loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        metrics=[RootMeanSquaredError()]
    )
    return model

def fit_weighted_model(model, xtrain, ytrain, sample_weights, epochs, batch_size, patience):
    """Treina o modelo com pesos"""
    
    # Callbacks avançados
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience,
        restore_best_weights=True, 
        min_delta=1e-6,
        mode='min'
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience//2,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [early_stop, reduce_lr]
    
    # Treina com pesos
    history = model.fit(
        xtrain, ytrain,
        sample_weight=sample_weights,
        epochs=epochs,
        validation_split=0.2,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def adaptive_prediction(model, last_sequence, scaler, n_predictions, confidence_threshold=0.1):
    """
    Predição adaptativa que ajusta com base na confiança
    """
    predictions = []
    confidences = []
    
    current_sequence = last_sequence.copy()
    
    for i in range(n_predictions):
        # Faz múltiplas predições para estimar incerteza
        pred_samples = []
        for _ in range(10):  # Monte Carlo Dropout
            pred_scaled = model.predict(current_sequence, verbose=0)
            pred_samples.append(pred_scaled[0, 0])
        
        # Calcula média e desvio padrão
        pred_mean = np.mean(pred_samples)
        pred_std = np.std(pred_samples)
        
        predictions.append(pred_mean)
        confidences.append(pred_std)
        
        # Atualiza sequência
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = pred_mean
    
    return np.array(predictions), np.array(confidences)

def smart_interpolation_fallback(data_imputed, missing_indices, method='cubic'):
    """
    Interpolação inteligente como fallback
    """
    for idx in missing_indices:
        if idx >= len(data_imputed):
            continue
            
        # Encontra valores válidos ao redor
        valid_indices = []
        valid_values = []
        
        # Busca em uma janela maior
        search_window = min(50, len(data_imputed)//4)
        
        for offset in range(1, search_window):
            # Verifica antes
            if idx - offset >= 0 and not np.isnan(data_imputed[idx - offset]):
                valid_indices.append(idx - offset)
                valid_values.append(data_imputed[idx - offset])
            
            # Verifica depois
            if idx + offset < len(data_imputed) and not np.isnan(data_imputed[idx + offset]):
                valid_indices.append(idx + offset)
                valid_values.append(data_imputed[idx + offset])
            
            if len(valid_indices) >= 4:  # Suficiente para interpolação cúbica
                break
        
        if len(valid_indices) >= 2:
            try:
                # Interpolação
                f = interp1d(valid_indices, valid_values, kind=method, 
                           bounds_error=False, fill_value='extrapolate')
                data_imputed[idx] = f(idx)
            except:
                # Fallback para interpolação linear
                if len(valid_indices) >= 2:
                    f = interp1d(valid_indices, valid_values, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
                    data_imputed[idx] = f(idx)
                else:
                    data_imputed[idx] = np.nanmean(valid_values)
        else:
            # Usa média global como último recurso
            data_imputed[idx] = np.nanmean(data_imputed)

def lstm_recursive_imputation_improved(
    data, missing_indices, look_back=5, units=64, learning_rate=0.001, 
    epochs=100, batch_size=32, patience=10, chunk_size=10, val_horizon=12,
    original_weight=2.0, imputed_weight=0.5, preprocessing_method='robust',
    use_adaptive_prediction=True, dropout_rate=0.2
):
    """
    Versão melhorada da imputação recursiva com sistema de pesos
    
    Args:
        original_weight: Peso para dados originais (maior = mais importante)
        imputed_weight: Peso para dados imputados (menor = menos importante)
        preprocessing_method: 'robust', 'standard' ou 'minmax'
        use_adaptive_prediction: Se usa predição adaptativa
        dropout_rate: Taxa de dropout para regularização
    """
    
    print(f"Iniciando imputação melhorada...")
    print(f"Peso dados originais: {original_weight}")
    print(f"Peso dados imputados: {imputed_weight}")
    print(f"Método de pré-processamento: {preprocessing_method}")
    
    # Copia os dados
    data_imputed = data.copy()
    training_history = []
    validation_metrics = []
    
    # Pré-processamento
    data_preprocessed, scaler = preprocess_data(data_imputed, preprocessing_method)
    
    # Cria array de pesos
    weights = np.full(len(data_imputed), original_weight)
    
    # Ordena os índices faltantes
    missing_indices_sorted = sorted(missing_indices)
    
    # Processa os valores faltantes em chunks
    for i in range(0, len(missing_indices_sorted), chunk_size):
        chunk_indices = missing_indices_sorted[i:i + chunk_size]
        
        print(f"\nProcessando chunk {i//chunk_size + 1}: índices {chunk_indices[0]} a {chunk_indices[-1]}")
        
        # Dados disponíveis até o chunk
        available_data_end = chunk_indices[0]
        available_data = data_imputed[:available_data_end]
        available_weights = weights[:available_data_end]
        
        # Remove NaNs
        valid_mask = ~np.isnan(available_data)
        available_data = available_data[valid_mask]
        available_weights = available_weights[valid_mask]
        
        if len(available_data) < look_back + 10:
            print(f"Dados insuficientes ({len(available_data)}). Usando interpolação...")
            smart_interpolation_fallback(data_imputed, chunk_indices)
            
            # Atualiza pesos dos valores imputados
            for idx in chunk_indices:
                if idx < len(weights):
                    weights[idx] = imputed_weight
            continue
        
        # Normalização
        available_data_reshaped = available_data.reshape(-1, 1)
        train_scaled = scaler.fit_transform(available_data_reshaped).flatten()
        
        # Cria dataset com pesos
        X_train, y_train, sample_weights = create_weighted_dataset(
            train_scaled, available_weights, look_back
        )
        
        if len(X_train) == 0:
            print("Não foi possível criar dataset")
            continue
        
        # Normaliza pesos
        sample_weights = sample_weights / np.mean(sample_weights)
        
        print(f"Treinando modelo com {len(X_train)} amostras...")
        print(f"Peso médio das amostras: {np.mean(sample_weights):.3f}")
        
        # Cria e treina modelo
        model = create_advanced_lstm(units, X_train.shape, learning_rate, dropout_rate)
        
        history = fit_weighted_model(
            model, X_train, y_train, sample_weights, 
            epochs, batch_size, patience
        )
        
        training_history.append(history)
        
        # Prepara sequência para predição
        last_sequence = train_scaled[-look_back:].reshape(1, look_back, 1)
        
        # Predição
        pred_horizon = val_horizon + len(chunk_indices)
        
        if use_adaptive_prediction:
            predictions_scaled, confidences = adaptive_prediction(
                model, last_sequence, scaler, pred_horizon
            )
        else:
            predictions_scaled = []
            for _ in range(pred_horizon):
                pred_scaled = model.predict(last_sequence, verbose=0)
                predictions_scaled.append(pred_scaled[0, 0])
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred_scaled[0, 0]
            predictions_scaled = np.array(predictions_scaled)
        
        # Desnormalização
        pred_array = predictions_scaled.reshape(-1, 1)
        predictions = scaler.inverse_transform(pred_array).flatten()
        
        # Separa predições
        val_preds = predictions[:val_horizon]
        chunk_preds = predictions[val_horizon:val_horizon + len(chunk_indices)]
        
        # Validação
        val_start_idx = available_data_end
        val_end_idx = min(val_start_idx + val_horizon, len(data_imputed))
        
        actual_vals = []
        pred_vals = []
        
        for j in range(val_horizon):
            current_idx = val_start_idx + j
            
            if current_idx >= val_end_idx:
                break
                
            if (not np.isnan(data_imputed[current_idx])) and (current_idx not in chunk_indices):
                actual_vals.append(data_imputed[current_idx])
                pred_vals.append(val_preds[j])
        
        # Calcula métricas
        if len(actual_vals) > 0:
            rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
            mae = np.mean(np.abs(np.array(pred_vals) - np.array(actual_vals)))
            data_range = np.max(actual_vals) - np.min(actual_vals)
            nrmse = rmse / data_range if data_range > 0 else np.nan
            
            # Calcula R² para avaliar ajuste
            ss_res = np.sum((np.array(actual_vals) - np.array(pred_vals)) ** 2)
            ss_tot = np.sum((np.array(actual_vals) - np.mean(actual_vals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            
            validation_metrics.append({
                'chunk': i//chunk_size + 1,
                'val_points': len(actual_vals),
                'rmse': rmse,
                'mae': mae,
                'nrmse': nrmse,
                'r2': r2,
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else np.nan
            })
            
            print(f"Validação no chunk {i//chunk_size + 1}:")
            print(f" - Pontos validados: {len(actual_vals)}/{val_horizon}")
            print(f" - RMSE: {rmse:.4f}")
            print(f" - MAE: {mae:.4f}")
            print(f" - NRMSE: {nrmse:.4f}")
            print(f" - R²: {r2:.4f}")
            print(f" - Loss final: {history.history['loss'][-1]:.6f}")
            
            if use_adaptive_prediction:
                avg_confidence = np.mean(confidences[:val_horizon])
                print(f" - Confiança média: {avg_confidence:.4f}")
        else:
            print(f"Validação no chunk {i//chunk_size + 1}: Nenhum ponto válido")
            validation_metrics.append({
                'chunk': i//chunk_size + 1,
                'val_points': 0,
                'rmse': np.nan,
                'mae': np.nan,
                'nrmse': np.nan,
                'r2': np.nan,
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else np.nan
            })
        
        # Aplica suavização nos valores imputados se a confiança for baixa
        if use_adaptive_prediction:
            chunk_confidences = confidences[val_horizon:val_horizon + len(chunk_indices)]
            high_uncertainty_mask = chunk_confidences > np.percentile(chunk_confidences, 75)
            
            if np.any(high_uncertainty_mask):
                print(f" - Aplicando suavização em {np.sum(high_uncertainty_mask)} predições de baixa confiança")
                # Suavização usando média móvel
                smoothed_preds = np.convolve(chunk_preds, np.ones(3)/3, mode='same')
                chunk_preds[high_uncertainty_mask] = smoothed_preds[high_uncertainty_mask]
        
        # Atualiza os valores do chunk
        for j, idx in enumerate(chunk_indices):
            if idx < len(data_imputed):
                data_imputed[idx] = chunk_preds[j]
                weights[idx] = imputed_weight  # Marca como imputado
                print(f"Imputado índice {idx}: {chunk_preds[j]:.4f}")
        
        # Limpeza de memória
        tf.keras.backend.clear_session()
    
    return data_imputed, training_history, validation_metrics

from itertools import product
import json
import gc

def score_from_validation_metrics(val_metrics, metric="rmse", reduction="mean"):
    """
    Converte a lista validation_metrics em um único escore.

    metric     : 'rmse' | 'mae' | 'nrmse' | 'r2'
    reduction  : 'mean' | 'median' | 'sum' | 'max'
    """
    vals = [m[metric] for m in val_metrics if not np.isnan(m[metric])]
    if not vals:      # nenhum ponto validado
        return np.inf if metric != "r2" else -np.inf
    if reduction == "mean":
        return float(np.mean(vals))
    if reduction == "median":
        return float(np.median(vals))
    if reduction == "sum":
        return float(np.sum(vals))
    if reduction == "max":
        return float(np.max(vals))
    raise ValueError("redução desconhecida")

def grid_search_imputation(
    data, 
    missing_indices,
    param_grid,
    metric="rmse",
    reduction="mean",
    verbose=True,
    save_results_path=None
):
    """
    Executa busca em grade nos hiperparâmetros do LSTM.

    param_grid = {
        'look_back':   [6, 8, 10],
        'units':       [64, 128],
        'learning_rate':[1e-3, 5e-4],
        'dropout_rate':[0.2, 0.3],
        'batch_size':  [16, 32],
        # parâmetros de lstm_recursive_imputation_improved
        # que NÃO mudarão ficam fixos logo abaixo
    }
    """
    results = []
    keys, values = zip(*param_grid.items())

    total = len(list(product(*values)))
    for run, combo in enumerate(product(*values), 1):
        params = dict(zip(keys, combo))

        if verbose:
            print(f"\n====== Grid-search {run}/{total} ======")
            print(json.dumps(params, indent=2))

        try:
            _, _, val_metrics = lstm_recursive_imputation_improved(
                data.copy(),               # sempre parte da série original
                missing_indices,
                **params,                  # hiperparâmetros que estamos variando
                # ---- parâmetros fixos que raramente mudam ----
                epochs=150,
                patience=15,
                chunk_size=8,
                val_horizon=20,
                original_weight=3.0,
                imputed_weight=0.3,
                preprocessing_method="robust",
                use_adaptive_prediction=True,
            )

            score = score_from_validation_metrics(
                val_metrics, metric=metric, reduction=reduction
            )

        except Exception as e:
            print(f"Falhou: {e}")
            score = np.inf if metric != "r2" else -np.inf

        results.append({**params, metric: score})

        # libera memória da GPU/TF
        tf.keras.backend.clear_session()
        gc.collect()

    df_results = pd.DataFrame(results).sort_values(metric, ascending=(metric!="r2"))
    if save_results_path:
        df_results.to_csv(save_results_path, index=False)
        if verbose:
            print(f"\nResultados salvos em {save_results_path}")

    return df_results

def evaluate_imputation_detailed(original_data, imputed_data, missing_indices):
    """Avaliação detalhada da imputação"""
    if len(missing_indices) == 0:
        return None
    
    true_values = original_data[missing_indices]
    predicted_values = imputed_data[missing_indices]
    
    # Remove NaNs se existirem
    valid_mask = ~(np.isnan(true_values) | np.isnan(predicted_values))
    true_values = true_values[valid_mask]
    predicted_values = predicted_values[valid_mask]
    
    if len(true_values) == 0:
        return None
    
    mae = np.mean(np.abs(predicted_values - true_values))
    rmse = np.sqrt(np.mean((predicted_values - true_values) ** 2))
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    
    data_range = np.max(true_values) - np.min(true_values)
    nrmse = rmse / data_range if data_range > 0 else np.nan
    
    # R²
    ss_res = np.sum((true_values - predicted_values) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Bias
    bias = np.mean(predicted_values - true_values)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'NRMSE': nrmse,
        'MAPE': mape,
        'R2': r2,
        'Bias': bias,
        'n_points': len(true_values)
    }

def plot_advanced_results(original_data, data_with_missing, imputed_data, missing_indices, 
                         validation_metrics=None, title="Resultados da Imputação Melhorada"):
    """Plot avançado dos resultados"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot principal
    ax1 = axes[0, 0]
    ax1.plot(range(len(original_data)), original_data, 'g-', label='Dados Originais', alpha=0.7, linewidth=1)
    
    available_mask = ~np.isnan(data_with_missing)
    ax1.plot(np.where(available_mask)[0], data_with_missing[available_mask], 'b-', label='Dados Disponíveis', linewidth=1.5)
    
    ax1.plot(missing_indices, imputed_data[missing_indices], 'ro', label='Valores Imputados', markersize=3)
    
    if len(missing_indices) > 0:
        ax1.axvspan(min(missing_indices), max(missing_indices), alpha=0.2, color='red', label='Área de Imputação')
    
    ax1.set_title(title)
    ax1.set_xlabel('Tempo')
    ax1.set_ylabel('Valor')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot de erros
    if validation_metrics:
        ax2 = axes[0, 1]
        chunks = [m['chunk'] for m in validation_metrics if not np.isnan(m['rmse'])]
        rmses = [m['rmse'] for m in validation_metrics if not np.isnan(m['rmse'])]
        maes = [m['mae'] for m in validation_metrics if not np.isnan(m['mae'])]
        
        if chunks:
            ax2.plot(chunks, rmses, 'ro-', label='RMSE', markersize=4)
            ax2.plot(chunks, maes, 'bo-', label='MAE', markersize=4)
            ax2.set_xlabel('Chunk')
            ax2.set_ylabel('Erro')
            ax2.set_title('Evolução dos Erros por Chunk')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # Histograma dos resíduos
    ax3 = axes[1, 0]
    if not np.all(np.isnan(original_data[missing_indices])):
        residuals = imputed_data[missing_indices] - original_data[missing_indices]
        residuals = residuals[~np.isnan(residuals)]
        if len(residuals) > 0:
            ax3.hist(residuals, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            ax3.axvline(np.mean(residuals), color='red', linestyle='--', label=f'Média: {np.mean(residuals):.3f}')
            ax3.set_xlabel('Resíduos')
            ax3.set_ylabel('Frequência')
            ax3.set_title('Distribuição dos Resíduos')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # Scatter plot: Predito vs Real
    ax4 = axes[1, 1]
    if not np.all(np.isnan(original_data[missing_indices])):
        true_vals = original_data[missing_indices]
        pred_vals = imputed_data[missing_indices]
        valid_mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
        
        if np.any(valid_mask):
            true_vals = true_vals[valid_mask]
            pred_vals = pred_vals[valid_mask]
            
            ax4.scatter(true_vals, pred_vals, alpha=0.6, color='blue', s=20)
            
            # Linha de referência y=x
            min_val = min(np.min(true_vals), np.min(pred_vals))
            max_val = max(np.max(true_vals), np.max(pred_vals))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax4.set_xlabel('Valores Reais')
            ax4.set_ylabel('Valores Preditos')
            ax4.set_title('Predito vs Real')
            ax4.grid(True, alpha=0.3)
            
            # Adiciona R²
            from scipy.stats import pearsonr
            if len(true_vals) > 1:
                r, p = pearsonr(true_vals, pred_vals)
                ax4.text(0.05, 0.95, f'R = {r:.3f}\np = {p:.3f}', 
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def uso_com_csv_melhorado(path_csv, **kwargs):
    """
    Versão melhorada para uso com CSV
    """
    df = pd.read_csv(path_csv)
    
    if 'Throughput' not in df.columns:
        raise ValueError("O CSV deve conter uma coluna chamada 'Throughput'.")
    
    original_data = df['Throughput'].values
    data_with_missing = original_data.copy()
    
    missing_indices = np.where(pd.isna(data_with_missing))[0]
    
    print("=== IMPUTAÇÃO LSTM MELHORADA ===")
    print(f"Total de valores faltantes: {len(missing_indices)}")
    print(f"Percentual faltante: {len(missing_indices)/len(data_with_missing)*100:.2f}%")
    
    # Parâmetros padrão melhorados
    default_params = {
        'look_back': 8,
        'units': 128,
        'learning_rate': 0.001,
        'epochs': 150,
        'batch_size': 16,
        'patience': 15,
        'chunk_size': 8,
        'val_horizon': 20,
        'original_weight': 3.0,
        'imputed_weight': 0.3,
        'preprocessing_method': 'robust',
        'use_adaptive_prediction': True,
        'dropout_rate': 0.3
    }
    
    # Sobrescreve com parâmetros fornecidos
    params = {**default_params, **kwargs}
    
    print(f"Parâmetros utilizados:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Executa imputação
    imputed_data, history, validation_metrics = lstm_recursive_imputation_improved(
        data_with_missing, missing_indices, **params
    )
    
    # Métricas de validação
    if validation_metrics:
        print("\n=== MÉTRICAS DE VALIDAÇÃO ===")
        rmse_list = []
        mae_list = []
        nrmse_list = []
        r2_list = []
        
        for item in validation_metrics:
            chunk = item['chunk']
            val_points = item['val_points']
            rmse = item.get('rmse', np.nan)
            mae = item.get('mae', np.nan)
            nrmse = item.get('nrmse', np.nan)
            r2 = item.get('r2', np.nan)
            
            print(f"Chunk {chunk}: {val_points} pts, RMSE={rmse:.4f}, MAE={mae:.4f}, NRMSE={nrmse:.4f}, R²={r2:.3f}")
            
            if not np.isnan(rmse):
                rmse_list.append(rmse)
            if not np.isnan(mae):
                mae_list.append(mae)
            if not np.isnan(nrmse):
                nrmse_list.append(nrmse)
            if not np.isnan(r2):
                r2_list.append(r2)
        
        if rmse_list:
            print(f"\n=== MÉDIAS FINAIS ===")
            print(f"RMSE médio: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
            print(f"MAE médio: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
            print(f"NRMSE médio: {np.mean(nrmse_list):.4f} ± {np.std(nrmse_list):.4f}")
            print(f"R² médio: {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")
    
    # Avaliação final
    evaluation = evaluate_imputation_detailed(original_data, imputed_data, missing_indices)
    if evaluation:
        print(f"\n=== AVALIAÇÃO FINAL ===")
        for metric, value in evaluation.items():
            print(f"{metric}: {value:.4f}") 
    # Estatísticas adicionais
    variance = np.var(original_data[~np.isnan(original_data)])
    missing_pct = len(missing_indices) / len(original_data) * 100

    print(f"\nEstatísticas dos dados:")
    print(f"Variância da série (sem NaNs): {variance:.4f}")
    print(f"Porcentagem de dados faltantes: {missing_pct:.2f}%")
   
    # Atualiza DataFrame e salva
    df['Throughput_imputed'] = imputed_data
    df.to_csv("imputed_output.csv", index=False)
    print("\nArquivo salvo como 'imputed_output.csv'")
    
    return df

# Função para aplicar em dados reais
def aplicar_imputacao_dados_reais(df, coluna_vazao, missing_indices, **kwargs):
    """
    Aplica imputação em dados reais
    
    Args:
        df: DataFrame com os dados
        coluna_vazao: nome da coluna com os valores de vazão
        missing_indices: índices dos valores faltantes
        **kwargs: parâmetros para lstm_recursive_imputation
    """
    
    # Extrai a série temporal
    data = df[coluna_vazao].values
    
    # Aplica imputação
    imputed_data, history = lstm_recursive_imputation_improved(data, missing_indices, **kwargs)
    
    # Atualiza DataFrame
    df_imputed = df.copy()
    df_imputed[coluna_vazao] = imputed_data
    
    return df_imputed, history

if __name__ == "__main__":
    # Executa exemplo
    uso_com_csv_melhorado("datasets/lowest failures treated/treated bbr esmond data ap-ba 07-03-2023.csv")

