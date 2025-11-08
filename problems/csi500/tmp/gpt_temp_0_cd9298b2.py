import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Acceleration with Regime-Aware Volume-Price Divergence
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    # Price Momentum Acceleration
    data['mom_accel_5d'] = (data['close'] / data['close'].shift(5) - 1) - (data['close'].shift(1) / data['close'].shift(6) - 1)
    data['mom_accel_10d'] = (data['close'] / data['close'].shift(10) - 1) - (data['close'].shift(1) / data['close'].shift(11) - 1)
    data['mom_accel_20d'] = (data['close'] / data['close'].shift(20) - 1) - (data['close'].shift(1) / data['close'].shift(21) - 1)
    
    # Volume-Price Divergence
    data['price_ret'] = data['close'].pct_change()
    data['volume_ret'] = data['volume'].pct_change()
    
    # Calculate rolling correlation for volume-price divergence
    vol_price_corr = data['price_ret'].rolling(window=10).corr(data['volume_ret'])
    data['vol_price_divergence'] = -vol_price_corr  # Negative correlation indicates divergence
    
    # Volume acceleration
    data['vol_accel_5d'] = (data['volume'] / data['volume'].shift(5) - 1) - (data['volume'].shift(1) / data['volume'].shift(6) - 1)
    data['accel_mismatch'] = data['mom_accel_5d'] - data['vol_accel_5d']
    
    # Regime Detection Using Amount Data
    data['amount_accel_5d'] = (data['amount'] / data['amount'].shift(5) - 1) - (data['amount'].shift(1) / data['amount'].shift(6) - 1)
    
    # Amount-based regime classification
    amount_ma = data['amount_accel_5d'].rolling(window=20).mean()
    amount_std = data['amount_accel_5d'].rolling(window=20).std()
    data['regime'] = np.where(data['amount_accel_5d'] > amount_ma + amount_std, 2,  # High participation
                     np.where(data['amount_accel_5d'] < amount_ma - amount_std, 0,  # Low participation
                             1))  # Normal participation
    
    # Volatility Context
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_20d'] = data['daily_range'].rolling(window=20).mean()
    
    # Exponential Smoothing Application
    alpha = 0.3
    data['smooth_mom_accel_5d'] = data['mom_accel_5d'].ewm(alpha=alpha).mean()
    data['smooth_mom_accel_10d'] = data['mom_accel_10d'].ewm(alpha=alpha).mean()
    data['smooth_mom_accel_20d'] = data['mom_accel_20d'].ewm(alpha=alpha).mean()
    data['smooth_vol_divergence'] = data['vol_price_divergence'].ewm(alpha=alpha).mean()
    data['smooth_accel_mismatch'] = data['accel_mismatch'].ewm(alpha=alpha).mean()
    
    # Trend persistence measurement
    data['mom_persistence_5d'] = data['smooth_mom_accel_5d'].rolling(window=5).apply(lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) == 5 else np.nan)
    data['mom_persistence_10d'] = data['smooth_mom_accel_10d'].rolling(window=5).apply(lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) == 5 else np.nan)
    
    # Cross-Sectional Ranking and Volatility Adjustment
    # Combined momentum-acceleration score
    data['combined_mom_accel'] = (data['smooth_mom_accel_5d'] * 0.4 + 
                                 data['smooth_mom_accel_10d'] * 0.3 + 
                                 data['smooth_mom_accel_20d'] * 0.3)
    
    # Cross-sectional ranking
    data['mom_accel_rank'] = data['combined_mom_accel'].rolling(window=20, min_periods=10).rank(pct=True)
    data['vol_div_rank'] = data['smooth_vol_divergence'].rolling(window=20, min_periods=10).rank(pct=True)
    
    # Volatility normalization
    data['volatility_scaled_mom'] = data['combined_mom_accel'] / (data['volatility_20d'] + 1e-8)
    
    # Adaptive Factor Construction
    # Signal combination with regime-dependent weighting
    def regime_adaptive_signal(row):
        if row['regime'] == 2:  # High participation - emphasize volume confirmation
            return (0.3 * row['smooth_mom_accel_5d'] + 
                   0.2 * row['smooth_mom_accel_10d'] + 
                   0.1 * row['smooth_mom_accel_20d'] + 
                   0.4 * row['smooth_vol_divergence'])
        elif row['regime'] == 0:  # Low participation - focus on momentum persistence
            return (0.4 * row['smooth_mom_accel_5d'] * row['mom_persistence_5d'] + 
                   0.3 * row['smooth_mom_accel_10d'] * row['mom_persistence_10d'] + 
                   0.3 * row['smooth_mom_accel_20d'])
        else:  # Normal participation - balanced approach
            return (0.35 * row['smooth_mom_accel_5d'] + 
                   0.25 * row['smooth_mom_accel_10d'] + 
                   0.2 * row['smooth_mom_accel_20d'] + 
                   0.2 * row['smooth_vol_divergence'])
    
    data['regime_adaptive_signal'] = data.apply(regime_adaptive_signal, axis=1)
    
    # Final volatility-scaled, cross-sectionally ranked output
    data['final_alpha'] = (data['regime_adaptive_signal'] * 
                          data['mom_accel_rank'] * 
                          data['vol_div_rank'] / 
                          (data['volatility_20d'] + 1e-8))
    
    return data['final_alpha']
