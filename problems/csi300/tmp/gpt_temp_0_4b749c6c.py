import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Scale Momentum Elasticity Analysis
    df['short_momentum'] = df['close'] / df['close'].shift(3) - 1
    df['medium_momentum'] = df['close'] / df['close'].shift(10) - 1
    df['long_momentum'] = df['close'] / df['close'].shift(20) - 1
    df['momentum_elasticity'] = (df['short_momentum'] * df['medium_momentum']) / (df['long_momentum'] + 1e-8)
    
    # Efficiency Divergence Analysis
    df['intraday_efficiency'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['gap_efficiency'] = (df['high'] - df['open']) / (abs(df['open'] - df['close'].shift(1)) + 1e-8)
    df['volume_weighted_efficiency'] = (df['close'] / df['close'].shift(5) - 1) * df['volume']
    df['pure_price_efficiency'] = df['close'] / df['close'].shift(5) - 1
    df['efficiency_divergence'] = (df['volume_weighted_efficiency'] - df['pure_price_efficiency']) * df['intraday_efficiency']
    
    # Dynamic Regime Classification
    df['close_returns'] = df['close'].pct_change()
    df['volatility_regime'] = df['close_returns'].rolling(window=5).std()
    
    # Calculate rolling correlation between intraday efficiency and volume
    efficiency_volume_corr = []
    for i in range(len(df)):
        if i >= 4:
            window_eff = df['intraday_efficiency'].iloc[i-4:i+1]
            window_vol = df['volume'].iloc[i-4:i+1]
            if len(window_eff) == 5 and len(window_vol) == 5:
                corr_val = window_eff.corr(window_vol)
                efficiency_volume_corr.append(corr_val if not np.isnan(corr_val) else 0)
            else:
                efficiency_volume_corr.append(0)
        else:
            efficiency_volume_corr.append(0)
    df['efficiency_volume_correlation'] = efficiency_volume_corr
    
    # Momentum persistence calculation
    momentum_persistence = []
    for i in range(len(df)):
        if i >= 2:
            current_sign = np.sign(df['close_returns'].iloc[i])
            count = 0
            for j in range(3):
                if i-j >= 0:
                    if np.sign(df['close_returns'].iloc[i-j]) == current_sign:
                        count += 1
                    else:
                        break
            momentum_persistence.append(count)
        else:
            momentum_persistence.append(1)
    df['momentum_persistence'] = momentum_persistence
    
    # Regime-Adaptive Signal Processing
    df['core_momentum_signal'] = df['momentum_elasticity'] * df['gap_efficiency']
    df['quality_adjusted_signal'] = df['core_momentum_signal'] * df['momentum_persistence']
    df['efficiency_weighted_signal'] = df['quality_adjusted_signal'] * df['efficiency_divergence']
    df['volatility_adjusted_signal'] = df['efficiency_weighted_signal'] / (df['volatility_regime'] + 1e-8)
    
    # Price Level Sensitivity Analysis
    df['price_20d_high'] = df['close'].rolling(window=20).max()
    df['price_20d_low'] = df['close'].rolling(window=20).min()
    df['price_range'] = df['price_20d_high'] - df['price_20d_low']
    df['price_position'] = (df['close'] - df['price_20d_low']) / (df['price_range'] + 1e-8)
    
    # Level multiplier based on price position quartiles
    df['level_multiplier'] = 1.0
    df.loc[df['price_position'] <= 0.25, 'level_multiplier'] = 0.8
    df.loc[df['price_position'] >= 0.75, 'level_multiplier'] = 1.2
    
    df['level_specific_signal'] = df['volatility_adjusted_signal'] * df['level_multiplier']
    
    # Composite Alpha Generation
    df['volume_20d_avg'] = df['volume'].rolling(window=20).mean()
    df['dynamic_liquidity_adjustment'] = df['volume'] / (df['volume_20d_avg'] + 1e-8)
    
    # Final alpha factor
    df['alpha_factor'] = df['level_specific_signal'] * df['dynamic_liquidity_adjustment']
    
    return df['alpha_factor']
