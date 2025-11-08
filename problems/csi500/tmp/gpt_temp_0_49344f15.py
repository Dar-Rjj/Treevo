import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum with Volume Confirmation alpha factor
    """
    df = data.copy()
    
    # Price Momentum Framework
    df['R5'] = df['close'] / df['close'].shift(5) - 1
    df['R10'] = df['close'] / df['close'].shift(10) - 1
    df['R20'] = df['close'] / df['close'].shift(20) - 1
    
    # Momentum Quality Assessment
    df['momentum_sign_agreement'] = ((df['R5'] > 0) & (df['R10'] > 0) & (df['R20'] > 0)) | \
                                   ((df['R5'] < 0) & (df['R10'] < 0) & (df['R20'] < 0))
    df['momentum_consistency_score'] = ((df['R5'] > 0).astype(int) + 
                                       (df['R10'] > 0).astype(int) + 
                                       (df['R20'] > 0).astype(int)) / 3
    df['momentum_strength'] = (abs(df['R5']) + abs(df['R10']) + abs(df['R20'])) / 3
    
    # Volatility Regime Classification
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Short-term volatility (10-day)
    df['vol_10d'] = returns.rolling(window=10).std()
    
    # Long-term volatility (30-day)
    df['vol_30d'] = returns.rolling(window=30).std()
    
    # Volatility ratio and regime determination
    df['vol_ratio'] = df['vol_10d'] / df['vol_30d']
    df['vol_regime'] = 'normal'
    df.loc[df['vol_ratio'] > 1.2, 'vol_regime'] = 'high'
    df.loc[df['vol_ratio'] < 0.8, 'vol_regime'] = 'low'
    
    # Volume Confirmation System
    # Volume momentum and stability
    df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_momentum'] = df['volume'] / df['volume_sma_5'].shift(1) - 1
    df['volume_stability'] = df['volume'] / df['volume_sma_20'].shift(1) - 1
    df['volume_divergence'] = df['volume_momentum'] - df['volume_stability']
    
    # Price-Volume Relationship
    # Calculate volume changes
    volume_changes = df['volume'].pct_change()
    
    # Recent correlation (10-day)
    recent_corr = []
    for i in range(len(df)):
        if i >= 10:
            recent_returns = returns.iloc[i-9:i+1].values
            recent_volume_changes = volume_changes.iloc[i-9:i+1].values
            if len(recent_returns) == 10 and len(recent_volume_changes) == 10:
                corr = np.corrcoef(recent_returns, recent_volume_changes)[0, 1]
                recent_corr.append(corr if not np.isnan(corr) else 0)
            else:
                recent_corr.append(0)
        else:
            recent_corr.append(0)
    df['recent_correlation'] = recent_corr
    
    # Historical correlation (30-day)
    historical_corr = []
    for i in range(len(df)):
        if i >= 30:
            hist_returns = returns.iloc[i-29:i+1].values
            hist_volume_changes = volume_changes.iloc[i-29:i+1].values
            if len(hist_returns) == 30 and len(hist_volume_changes) == 30:
                corr = np.corrcoef(hist_returns, hist_volume_changes)[0, 1]
                historical_corr.append(corr if not np.isnan(corr) else 0)
            else:
                historical_corr.append(0)
        else:
            historical_corr.append(0)
    df['historical_correlation'] = historical_corr
    
    # Alpha Factor Integration
    # Regime-Weighted Momentum
    df['regime_weighted_momentum'] = 0.0
    
    # High volatility regime weights
    high_vol_mask = df['vol_regime'] == 'high'
    df.loc[high_vol_mask, 'regime_weighted_momentum'] = (
        0.6 * df.loc[high_vol_mask, 'R5'] + 
        0.3 * df.loc[high_vol_mask, 'R10'] + 
        0.1 * df.loc[high_vol_mask, 'R20']
    )
    
    # Low volatility regime weights
    low_vol_mask = df['vol_regime'] == 'low'
    df.loc[low_vol_mask, 'regime_weighted_momentum'] = (
        0.2 * df.loc[low_vol_mask, 'R5'] + 
        0.3 * df.loc[low_vol_mask, 'R10'] + 
        0.5 * df.loc[low_vol_mask, 'R20']
    )
    
    # Normal volatility regime weights
    normal_vol_mask = df['vol_regime'] == 'normal'
    df.loc[normal_vol_mask, 'regime_weighted_momentum'] = (
        0.3 * df.loc[normal_vol_mask, 'R5'] + 
        0.4 * df.loc[normal_vol_mask, 'R10'] + 
        0.3 * df.loc[normal_vol_mask, 'R20']
    )
    
    # Volume Confirmation Multiplier
    df['volume_confirmation_multiplier'] = (
        df['volume_divergence'] * df['momentum_consistency_score']
    )
    
    # Final Alpha Construction
    df['base_factor'] = df['regime_weighted_momentum'] * df['volume_confirmation_multiplier']
    
    # Correlation Adjustment
    df['final_alpha'] = df['base_factor']
    negative_corr_mask = df['recent_correlation'] <= 0
    df.loc[negative_corr_mask, 'final_alpha'] = -df.loc[negative_corr_mask, 'final_alpha']
    
    # Momentum Quality Filter
    low_consistency_mask = df['momentum_consistency_score'] <= 0.5
    df.loc[low_consistency_mask, 'final_alpha'] = 0
    
    return df['final_alpha']
