import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Momentum Efficiency Factor
    Combines fractal analysis, multi-scale momentum, volume efficiency, and regime detection
    """
    data = df.copy()
    
    # 1. Calculate Fractal-Adjusted Momentum
    # Compute Price Fractal Dimension (Hurst exponent approximation)
    def hurst_exponent(series, window=10):
        lags = range(2, min(window, len(series))//2)
        tau = []
        for lag in lags:
            if len(series) >= lag:
                tau.append(np.std(np.subtract(series[lag:], series[:-lag])))
        if len(tau) < 2:
            return 0.5
        try:
            hurst = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)[0]
            return max(0.1, min(0.9, hurst))
        except:
            return 0.5
    
    # Calculate fractal dimension for each day
    fractal_dim = []
    for i in range(len(data)):
        if i < 20:
            fractal_dim.append(0.5)
            continue
        window_data = data.iloc[i-19:i+1]
        high_series = window_data['high'].values
        low_series = window_data['low'].values
        close_series = window_data['close'].values
        
        # Combine price information for fractal analysis
        price_combined = (high_series + low_series + close_series) / 3
        fractal_dim.append(hurst_exponent(price_combined, 10))
    
    data['fractal_dim'] = fractal_dim
    
    # Multi-Scale Volatility-Adjusted Momentum
    data['volatility'] = (data['high'] - data['low']) / data['close']
    data['volatility'] = data['volatility'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Short-term momentum (5-day)
    data['momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_5d_adj'] = data['momentum_5d'] / np.maximum(data['volatility'], 0.001)
    
    # Medium-term momentum (10-day)
    data['momentum_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['momentum_10d_adj'] = data['momentum_10d'] / np.maximum(data['volatility'], 0.001)
    
    # Long-term momentum (20-day)
    data['momentum_20d'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['momentum_20d_adj'] = data['momentum_20d'] / np.maximum(data['volatility'], 0.001)
    
    # Fractal-weighted momentum combination
    def fractal_weighted_momentum(row):
        fd = row['fractal_dim']
        if fd > 0.7:  # High fractal dimension - emphasize short-term
            return 0.6 * row['momentum_5d_adj'] + 0.3 * row['momentum_10d_adj'] + 0.1 * row['momentum_20d_adj']
        elif fd < 0.3:  # Low fractal dimension - emphasize long-term
            return 0.1 * row['momentum_5d_adj'] + 0.3 * row['momentum_10d_adj'] + 0.6 * row['momentum_20d_adj']
        else:  # Medium fractal dimension - balanced
            return 0.33 * row['momentum_5d_adj'] + 0.34 * row['momentum_10d_adj'] + 0.33 * row['momentum_20d_adj']
    
    data['fractal_momentum'] = data.apply(fractal_weighted_momentum, axis=1)
    
    # 2. Analyze Volume-Confluence Efficiency
    # Price Range Efficiency
    data['range_efficiency'] = (data['close'] - data['open']) / np.maximum(data['high'] - data['low'], 0.001)
    
    # Volume Distribution Patterns (5-day rolling)
    data['volume_ma_5d'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_concentration'] = data['volume'] / np.maximum(data['volume_ma_5d'], 1)
    
    # Volume-Volatility Correlation (6-day rolling)
    def rolling_volume_vol_corr(df, window=6):
        correlations = []
        for i in range(len(df)):
            if i < window:
                correlations.append(0)
            else:
                vol_window = df['volume'].iloc[i-window+1:i+1]
                range_window = (df['high'] - df['low']).iloc[i-window+1:i+1]
                if len(vol_window) >= 2 and len(range_window) >= 2:
                    try:
                        corr = vol_window.corr(range_window)
                        correlations.append(0 if pd.isna(corr) else corr)
                    except:
                        correlations.append(0)
                else:
                    correlations.append(0)
        return correlations
    
    data['volume_vol_corr'] = rolling_volume_vol_corr(data, 6)
    
    # Volume Efficiency Score
    data['volume_efficiency'] = (
        0.4 * data['range_efficiency'].fillna(0) +
        0.3 * data['volume_concentration'].fillna(0) +
        0.3 * data['volume_vol_corr'].fillna(0)
    )
    
    # 3. Generate Adaptive Regime Signals
    # Bidirectional Pressure
    data['upward_pressure'] = (data['high'] - data['close']).rolling(window=3, min_periods=1).mean()
    data['downward_pressure'] = (data['close'] - data['low']).rolling(window=3, min_periods=1).mean()
    data['pressure_imbalance'] = (data['upward_pressure'] - data['downward_pressure']) / np.maximum(
        data['upward_pressure'] + data['downward_pressure'], 0.001
    )
    
    # ATR-based Market Regime Classification
    data['atr'] = (
        (data['high'] - data['low']).rolling(window=14, min_periods=1).mean() / data['close']
    )
    data['atr_ma'] = data['atr'].rolling(window=20, min_periods=1).mean()
    
    def classify_regime(row):
        atr_ratio = row['atr'] / max(row['atr_ma'], 0.001)
        if atr_ratio > 1.2:
            return 'trending'  # High volatility - trending market
        elif atr_ratio < 0.8:
            return 'range_bound'  # Low volatility - range-bound market
        else:
            return 'transition'  # Medium volatility - transition phase
    
    data['regime'] = data.apply(classify_regime, axis=1)
    
    # Regime-Adaptive Combination
    def regime_adaptive_factor(row):
        fractal_mom = row['fractal_momentum']
        vol_eff = row['volume_efficiency']
        pressure = row['pressure_imbalance']
        
        if row['regime'] == 'trending':
            # Emphasize momentum with volume confirmation
            return 0.6 * fractal_mom + 0.3 * vol_eff + 0.1 * pressure
        elif row['regime'] == 'range_bound':
            # Emphasize pressure imbalance with fractal weighting
            return 0.2 * fractal_mom + 0.2 * vol_eff + 0.6 * pressure
        else:  # transition
            # Balance momentum and efficiency signals
            return 0.4 * fractal_mom + 0.4 * vol_eff + 0.2 * pressure
    
    data['regime_factor'] = data.apply(regime_adaptive_factor, axis=1)
    
    # 4. Final Factor Synthesis
    def final_factor_synthesis(row):
        fractal_mom = row['fractal_momentum']
        vol_eff = row['volume_efficiency']
        regime_factor = row['regime_factor']
        vol_vol_corr = row['volume_vol_corr']
        
        # High efficiency + confirming volume + appropriate fractal momentum
        efficiency_score = abs(vol_eff)
        momentum_strength = abs(fractal_mom)
        
        if efficiency_score > 0.7 and vol_vol_corr > 0.5:
            # Strong directional bias
            factor_strength = 1.2
        elif efficiency_score < 0.3 and vol_vol_corr < -0.3:
            # Reversal potential
            factor_strength = 0.8
        else:
            # Mixed signals with volume-volatility confluence
            factor_strength = 1.0
        
        return regime_factor * factor_strength * np.sign(fractal_mom)
    
    data['final_factor'] = data.apply(final_factor_synthesis, axis=1)
    
    # Clean and return the factor
    factor = data['final_factor'].replace([np.inf, -np.inf], 0).fillna(0)
    return factor
