import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Divergence Analysis
    # Short-term divergence: (Close_t / Close_t-5 - 1) - (Volume_t / Volume_t-5 - 1)
    short_term_div = (data['close'] / data['close'].shift(5) - 1) - (data['volume'] / data['volume'].shift(5) - 1)
    
    # Medium-term divergence: (Close_t / Close_t-10 - 1) - (Volume_t / Volume_t-10 - 1)
    medium_term_div = (data['close'] / data['close'].shift(10) - 1) - (data['volume'] / data['volume'].shift(10) - 1)
    
    # Divergence acceleration: (short-term divergence - medium-term divergence) / |medium-term divergence|
    div_acceleration = (short_term_div - medium_term_div) / np.abs(medium_term_div.replace(0, np.nan))
    
    # Divergence persistence: Count(divergence_t > divergence_t-1 for past 3 days) / 3
    div_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(3, len(data)):
        window_div = short_term_div.iloc[i-3:i+1]
        persistence_count = sum(window_div.iloc[j] > window_div.iloc[j-1] for j in range(1, len(window_div)))
        div_persistence.iloc[i] = persistence_count / 3
    
    # Order Flow Imbalance Detection
    # Intraday pressure imbalance: (Close_t - Open_t) / (High_t - Low_t)
    intraday_pressure = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume concentration: Volume_t / (Σ Volume_t-4:t)
    volume_concentration = data['volume'] / data['volume'].rolling(window=5).sum()
    
    # Price impact efficiency: |Close_t - Open_t| / Volume_t
    price_impact_efficiency = np.abs(data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    
    # Microstructure noise: (High_t - Low_t) / Close_t
    microstructure_noise = (data['high'] - data['low']) / data['close']
    
    # Volatility Structure Analysis
    # Volatility clustering: std(Close_t-4:t) / std(Close_t-9:t-5)
    vol_clustering = data['close'].rolling(window=5).std() / data['close'].shift(5).rolling(window=5).std()
    
    # Range volatility ratio: (High_t - Low_t) / std(Close_t-4:t)
    range_vol_ratio = (data['high'] - data['low']) / data['close'].rolling(window=5).std().replace(0, np.nan)
    
    # Gap volatility: |Open_t - Close_t-1| / (High_t - Low_t)
    gap_volatility = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volatility persistence: std(Close_t-4:t) / std(Close_t-9:t-5)
    vol_persistence = data['close'].rolling(window=5).std() / data['close'].shift(5).rolling(window=5).std()
    
    # Market Regime Classification
    # Trend strength: Close_t / MA(Close, 10)_t-1 - 1
    ma_10 = data['close'].rolling(window=10).mean().shift(1)
    trend_strength = data['close'] / ma_10 - 1
    
    # Mean-reversion potential: |Close_t - MA(Close, 10)_t-1| / std(Close_t-9:t)
    mean_reversion_potential = np.abs(data['close'] - ma_10) / data['close'].rolling(window=10).std().replace(0, np.nan)
    
    # Regime determination
    trending_regime = trend_strength > (2 * mean_reversion_potential)
    mean_reverting_regime = ~trending_regime
    
    # Divergence-Microstructure Integration
    # Volume confirmation: divergence persistence × volume concentration
    volume_confirmation = div_persistence * volume_concentration
    
    # Price confirmation: divergence acceleration × intraday pressure imbalance
    price_confirmation = div_acceleration * intraday_pressure
    
    # Regime-Adaptive Alpha Generation
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Trending regime processing
    trending_alpha = (div_acceleration * volume_concentration * vol_persistence * price_impact_efficiency)
    
    # Mean-reverting regime processing
    mean_reverting_alpha = (div_persistence * microstructure_noise * gap_volatility * intraday_pressure)
    
    # Unified alpha synthesis
    for i in range(len(data)):
        if trending_regime.iloc[i]:
            alpha.iloc[i] = trending_alpha.iloc[i]
        else:
            alpha.iloc[i] = mean_reverting_alpha.iloc[i]
    
    # Multiply by absolute divergence acceleration for signal strength
    alpha = alpha * np.abs(div_acceleration)
    
    # Incorporate volatility clustering for adaptive scaling
    alpha = alpha * vol_clustering
    
    return alpha
