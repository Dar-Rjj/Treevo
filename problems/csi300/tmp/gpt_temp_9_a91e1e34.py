import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Momentum Efficiency with Volume-Range Divergence factor
    """
    # Calculate returns
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    # Multi-timeframe volatility calculation
    df['vol_3d'] = df['returns'].rolling(window=3).std()
    df['vol_5d'] = df['returns'].rolling(window=5).std()
    df['vol_5d_avg'] = df['vol_5d'].rolling(window=5).mean()
    
    # Volatility regime classification
    conditions = [
        df['vol_5d'] > 1.5 * df['vol_5d_avg'],
        df['vol_5d'] < 0.7 * df['vol_5d_avg']
    ]
    choices = [2, 0]  # 2: High, 0: Low, 1: Normal
    df['vol_regime'] = np.select(conditions, choices, default=1)
    
    # Regime transition analysis
    df['regime_shift'] = df['vol_regime'].diff()
    df['regime_persistence'] = df.groupby((df['vol_regime'] != df['vol_regime'].shift()).cumsum()).cumcount() + 1
    
    # Volatility jump magnitude
    df['vol_jump'] = (df['vol_5d'] / df['vol_5d'].shift() - 1).fillna(0)
    
    # Intraday volatility patterns
    df['intraday_range'] = (df['high'] - df['low']) / df['close']
    df['overnight_return'] = (df['open'] - df['close'].shift()) / df['close'].shift()
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['overnight_vol_ratio'] = abs(df['overnight_return']) / (abs(df['intraday_return']) + 1e-8)
    
    # Range efficiency calculation
    df['daily_return'] = abs(df['returns'])
    df['range_efficiency'] = df['daily_return'] / (df['intraday_range'] + 1e-8)
    
    # Multi-timeframe efficiency analysis
    df['eff_5d'] = df['range_efficiency'].rolling(window=5).mean()
    df['eff_10d'] = df['range_efficiency'].rolling(window=10).mean()
    df['efficiency_ratio'] = df['eff_5d'] / (df['eff_10d'] + 1e-8)
    
    # Momentum calculations
    df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_6d'] = df['close'] / df['close'].shift(6) - 1
    df['momentum_acceleration'] = df['momentum_3d'] / (df['momentum_6d'] + 1e-8)
    
    # Volume analysis
    df['volume_5d_avg'] = df['volume'].rolling(window=5).mean()
    df['volume_gradient'] = df['volume'] / df['volume'].shift() - 1
    df['volume_persistence'] = df['volume'] / df['volume'].shift(3) - 1
    
    # Volume regime classification
    vol_conditions = [
        df['volume'] > 1.3 * df['volume_5d_avg'],
        df['volume'] < 0.8 * df['volume_5d_avg']
    ]
    df['volume_regime'] = np.select(vol_conditions, choices, default=1)
    
    # Volume-range divergence
    df['range_expansion'] = df['intraday_range'] / df['intraday_range'].shift() - 1
    df['volume_range_divergence'] = df['volume_gradient'] - df['range_expansion']
    
    # Amount-weighted signals
    df['amount_5d_avg'] = df['amount'].rolling(window=5).mean()
    df['amount_intensity'] = df['amount'] / (df['amount_5d_avg'] + 1e-8)
    df['amount_weighted_momentum'] = df['momentum_3d'] * df['amount_intensity']
    
    # Regime-specific calculations
    def regime_specific_calc(df, col, regime_col='vol_regime'):
        """Calculate regime-specific statistics"""
        results = []
        for regime in [0, 1, 2]:  # Low, Normal, High
            mask = df[regime_col] == regime
            if mask.any():
                regime_vals = df.loc[mask, col]
                if len(regime_vals) > 0:
                    results.append(regime_vals.iloc[-1] if len(regime_vals) > 0 else 0)
                else:
                    results.append(0)
            else:
                results.append(0)
        return results
    
    # Calculate regime-specific efficiency
    regime_efficiency = []
    for i in range(len(df)):
        if i >= 5:  # Ensure enough data
            current_regime = df['vol_regime'].iloc[i]
            regime_data = df.iloc[max(0, i-4):i+1]
            regime_data = regime_data[regime_data['vol_regime'] == current_regime]
            if len(regime_data) > 0:
                regime_efficiency.append(regime_data['range_efficiency'].mean())
            else:
                regime_efficiency.append(df['range_efficiency'].iloc[i])
        else:
            regime_efficiency.append(df['range_efficiency'].iloc[i])
    
    df['regime_efficiency'] = regime_efficiency
    
    # Regime-weighted momentum efficiency
    regime_weights = {0: 1.2, 1: 1.0, 2: 0.8}  # Higher weight for low volatility regimes
    df['regime_weight'] = df['vol_regime'].map(regime_weights)
    df['weighted_momentum_efficiency'] = df['momentum_3d'] * df['regime_efficiency'] * df['regime_weight']
    
    # Transition signals
    df['transition_signal'] = 0
    # Low to High transition
    low_to_high = (df['vol_regime'].shift() == 0) & (df['vol_regime'] == 2)
    df.loc[low_to_high, 'transition_signal'] = df.loc[low_to_high, 'momentum_3d'] * df.loc[low_to_high, 'efficiency_ratio']
    
    # High to Low transition  
    high_to_low = (df['vol_regime'].shift() == 2) & (df['vol_regime'] == 0)
    df.loc[high_to_low, 'transition_signal'] = df.loc[high_to_low, 'momentum_6d'] * df.loc[high_to_low, 'regime_efficiency']
    
    # Volume-range divergence confirmation
    df['divergence_confirmation'] = df['volume_range_divergence'] * df['amount_weighted_momentum']
    
    # Final alpha factor construction
    df['alpha_factor'] = (
        df['weighted_momentum_efficiency'] * 0.4 +
        df['transition_signal'] * 0.3 +
        df['divergence_confirmation'] * 0.2 +
        df['regime_persistence'] * df['momentum_acceleration'] * 0.1
    )
    
    # Apply regime memory decay
    regime_decay = {0: 0.95, 1: 0.9, 2: 0.85}  # Higher decay for high volatility
    df['regime_decay'] = df['vol_regime'].map(regime_decay)
    
    # Smooth factor with regime-appropriate decay
    alpha_smoothed = []
    for i in range(len(df)):
        if i == 0:
            alpha_smoothed.append(df['alpha_factor'].iloc[i])
        else:
            decay = df['regime_decay'].iloc[i]
            smoothed = decay * alpha_smoothed[-1] + (1 - decay) * df['alpha_factor'].iloc[i]
            alpha_smoothed.append(smoothed)
    
    df['alpha_smoothed'] = alpha_smoothed
    
    return df['alpha_smoothed']
