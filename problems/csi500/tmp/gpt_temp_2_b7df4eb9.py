import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Momentum Convergence Analysis
    # Short-Term Momentum Component
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['vol_accel_3d'] = data['volume'] / ((data['volume'].shift(3) + data['volume'].shift(2) + data['volume'].shift(1)) / 3) - 1
    
    # Medium-Term Momentum Component
    data['mom_20d'] = data['close'] / data['close'].shift(20) - 1
    vol_10d_avg = data['volume'].rolling(window=10, min_periods=10).mean().shift(1)
    data['vol_trend_10d'] = data['volume'] / vol_10d_avg - 1
    
    # Momentum Convergence Signal
    data['price_mom_conv'] = data['mom_5d'] - data['mom_20d']
    data['vol_mom_conv'] = data['vol_accel_3d'] - data['vol_trend_10d']
    data['momentum_convergence'] = data['price_mom_conv'] * data['vol_mom_conv']
    
    # Volatility Regime Classification & Adaptation
    # True Range Calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['tr_5d_avg'] = data['tr'].rolling(window=5, min_periods=5).mean()
    
    # Volatility Regime Classification
    conditions = [
        data['tr'] > 1.2 * data['tr_5d_avg'],
        data['tr'] < 0.8 * data['tr_5d_avg']
    ]
    choices = ['high', 'low']
    data['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Regime-Adaptive Momentum Weighting
    def get_momentum_weight(row):
        if row['vol_regime'] == 'high':
            return 0.7 * row['mom_5d'] + 0.3 * row['mom_20d']
        elif row['vol_regime'] == 'low':
            return 0.3 * row['mom_5d'] + 0.7 * row['mom_20d']
        else:
            return 0.5 * row['mom_5d'] + 0.5 * row['mom_20d']
    
    data['regime_weighted_mom'] = data.apply(get_momentum_weight, axis=1)
    
    # Volatility Persistence Adjustment
    regime_changes = data['vol_regime'] != data['vol_regime'].shift(1)
    data['regime_group'] = regime_changes.cumsum()
    data['consecutive_days'] = data.groupby('regime_group').cumcount() + 1
    data['persistence_multiplier'] = 1 + 0.1 * np.minimum(data['consecutive_days'], 5)
    
    # Price-Volume Entropy Efficiency Analysis
    # Directional Efficiency
    data['abs_price_change'] = abs(data['close'] - data['prev_close'])
    data['price_efficiency'] = data['abs_price_change'] / data['tr']
    
    # Volume Directional Alignment
    data['return_sign'] = np.sign(data['close'] - data['prev_close'])
    data['vol_accel_sign'] = np.sign(data['vol_accel_3d'])
    data['vol_alignment'] = data['return_sign'] * data['vol_accel_sign']
    
    # Entropy-Based Uncertainty Measurement
    # Volume Distribution Skewness
    data['vol_skew_5d'] = data['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: ((x - x.mean())**3).mean() / (x.std()**3 + 1e-10), raw=True
    )
    
    # Price-Volume Correlation Entropy
    def calculate_corr_entropy(series):
        if len(series) < 10:
            return np.nan
        returns = series['close'].pct_change().dropna()
        volumes = series['volume'].iloc[1:]
        if len(returns) < 9:
            return np.nan
        corr = returns.abs().corr(volumes)
        return -corr * np.log(abs(corr) + 1e-10)
    
    rolling_data = [data[['close', 'volume']].iloc[i-9:i+1] for i in range(len(data))]
    data['pv_corr_entropy'] = [calculate_corr_entropy(chunk) if len(chunk) == 10 else np.nan for chunk in rolling_data]
    
    # Normalize entropy to [0,1] range
    entropy_rolling = data['pv_corr_entropy'].rolling(window=20, min_periods=20)
    data['entropy_min'] = entropy_rolling.min()
    data['entropy_max'] = entropy_rolling.max()
    data['norm_entropy'] = (data['pv_corr_entropy'] - data['entropy_min']) / (data['entropy_max'] - data['entropy_min'] + 1e-10)
    
    # Efficiency-Entropy Composite
    data['efficiency_entropy'] = data['price_efficiency'] * (1 - data['norm_entropy']) * (1 + 0.5 * data['vol_skew_5d'])
    
    # Final Composite Alpha Generation
    # Combine Momentum Convergence with Efficiency-Entropy
    data['composite_signal'] = data['momentum_convergence'] * data['efficiency_entropy'] * data['persistence_multiplier']
    
    # Volatility Scaling & Risk Adjustment
    def get_volatility_measure(row):
        if row['vol_regime'] == 'high':
            returns = data['close'].pct_change().rolling(window=3, min_periods=3).std()
            return returns.loc[row.name] if row.name in returns.index else np.nan
        elif row['vol_regime'] == 'low':
            returns = data['close'].pct_change().rolling(window=10, min_periods=10).std()
            return returns.loc[row.name] if row.name in returns.index else np.nan
        else:
            returns = data['close'].pct_change().rolling(window=5, min_periods=5).std()
            return returns.loc[row.name] if row.name in returns.index else np.nan
    
    data['volatility_measure'] = data.apply(get_volatility_measure, axis=1)
    
    # Final factor with volatility scaling
    data['final_factor'] = data['composite_signal'] / (data['volatility_measure'] + 1e-10)
    
    return data['final_factor']
