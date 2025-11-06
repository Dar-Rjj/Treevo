import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Asymmetry-Volatility Momentum Convergence factor
    """
    df = df.copy()
    
    # 1. Asymmetric Volume-Price Momentum Analysis
    # Calculate returns and volume changes
    df['returns'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Directional volume momentum patterns
    # Up-day and down-day volume averages (4-day window)
    df['up_day'] = df['returns'] > 0
    df['down_day'] = df['returns'] < 0
    
    # Rolling volume averages for up and down days
    up_volume_avg = []
    down_volume_avg = []
    
    for i in range(len(df)):
        if i < 4:
            up_volume_avg.append(np.nan)
            down_volume_avg.append(np.nan)
            continue
            
        window = df.iloc[i-4:i]
        up_days = window[window['up_day']]
        down_days = window[window['down_day']]
        
        up_avg = up_days['volume'].mean() if len(up_days) > 0 else np.nan
        down_avg = down_days['volume'].mean() if len(down_days) > 0 else np.nan
        
        up_volume_avg.append(up_avg)
        down_volume_avg.append(down_avg)
    
    df['up_volume_avg'] = up_volume_avg
    df['down_volume_avg'] = down_volume_avg
    
    # Volume momentum components
    df['up_volume_momentum'] = (df['volume'] / df['up_volume_avg']) * (df['close'] - df['close'].shift(1))
    df['down_volume_momentum'] = (df['volume'] / df['down_volume_avg']) * (df['close'] - df['close'].shift(1))
    df['volume_asymmetry_momentum'] = df['up_volume_momentum'] - df['down_volume_momentum']
    
    # Multi-timeframe volume-price divergence
    df['price_momentum_3d'] = df['close'].pct_change(3)
    df['volume_momentum_3d'] = df['volume'].pct_change(3)
    df['short_term_divergence'] = df['price_momentum_3d'] - df['volume_momentum_3d']
    
    df['price_momentum_5d'] = df['close'].pct_change(5)
    df['volume_momentum_5d'] = df['volume'].pct_change(5)
    df['medium_term_divergence'] = df['price_momentum_5d'] - df['volume_momentum_5d']
    
    df['price_momentum_10d'] = df['close'].pct_change(10)
    df['volume_momentum_10d'] = df['volume'].pct_change(10)
    df['long_term_divergence'] = df['price_momentum_10d'] - df['volume_momentum_10d']
    
    # 2. Volatility-Adaptive Asymmetry Framework
    # True Range Calculation
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility measures
    df['atr_5'] = df['true_range'].rolling(window=5).mean() / df['close']
    df['atr_20'] = df['true_range'].rolling(window=20).mean() / df['close']
    df['volatility_ratio'] = df['atr_5'] / df['atr_20']
    
    # Volatility-scaled asymmetry momentum
    df['high_vol_asymmetry'] = df['volume_asymmetry_momentum'].rolling(window=3).mean() / (df['atr_5'] + 1e-8)
    df['normal_vol_asymmetry'] = df['volume_asymmetry_momentum'].rolling(window=5).mean()
    
    # 3. Multi-Scale Momentum Convergence Engine
    # Price momentum across timeframes
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_20d'] = df['close'].pct_change(20)
    
    # Volume-weighted momentum
    df['volume_momentum_5d'] = df['volume'].pct_change(5)
    df['volume_momentum_10d'] = df['volume'].pct_change(10)
    df['volume_momentum_20d'] = df['volume'].pct_change(20)
    
    # Volume-confirmed momentum
    df['volume_confirmed_5d'] = df['momentum_5d'] * (1 + df['volume_momentum_5d'])
    df['volume_confirmed_10d'] = df['momentum_10d'] * (1 + df['volume_momentum_10d'])
    df['volume_confirmed_20d'] = df['momentum_20d'] * (1 + df['volume_momentum_20d'])
    
    # 4. Regime-Dependent Factor Synthesis
    # Volatility regime classification
    df['vol_regime'] = 'normal'
    df.loc[df['volatility_ratio'] > 1.3, 'vol_regime'] = 'high'
    df.loc[df['volatility_ratio'] < 0.8, 'vol_regime'] = 'low'
    
    # Asymmetry regime detection
    asymmetry_threshold = df['volume_asymmetry_momentum'].abs().rolling(window=10).mean()
    df['asymmetry_regime'] = 'symmetry'
    df.loc[df['volume_asymmetry_momentum'].abs() > 2 * asymmetry_threshold, 'asymmetry_regime'] = 'strong'
    df.loc[(df['volume_asymmetry_momentum'].abs() > asymmetry_threshold) & 
           (df['volume_asymmetry_momentum'].abs() <= 2 * asymmetry_threshold), 'asymmetry_regime'] = 'weak'
    
    # Dynamic alpha generation based on regimes
    alpha_components = []
    
    for i in range(len(df)):
        if pd.isna(df.iloc[i]['vol_regime']) or pd.isna(df.iloc[i]['asymmetry_regime']):
            alpha_components.append(np.nan)
            continue
            
        vol_regime = df.iloc[i]['vol_regime']
        asym_regime = df.iloc[i]['asymmetry_regime']
        
        if vol_regime == 'high' and asym_regime == 'strong':
            # Focus on short-term divergence acceleration
            component = (df.iloc[i]['short_term_divergence'] * 
                        df.iloc[i]['high_vol_asymmetry'] * 
                        df.iloc[i]['momentum_5d'])
        elif vol_regime == 'low' and asym_regime == 'weak':
            # Emphasize medium-term convergence breaks
            component = (df.iloc[i]['medium_term_divergence'] * 
                        df.iloc[i]['normal_vol_asymmetry'] * 
                        df.iloc[i]['momentum_10d'])
        elif vol_regime == 'normal' and asym_regime == 'symmetry':
            # Multi-scale momentum alignment
            component = (df.iloc[i]['volume_confirmed_5d'] + 
                        df.iloc[i]['volume_confirmed_10d'] + 
                        df.iloc[i]['volume_confirmed_20d']) / 3
        else:
            # Transition regimes - weighted reversal confirmation
            component = (0.4 * df.iloc[i]['short_term_divergence'] + 
                        0.3 * df.iloc[i]['medium_term_divergence'] + 
                        0.3 * df.iloc[i]['long_term_divergence'])
        
        alpha_components.append(component)
    
    df['regime_alpha'] = alpha_components
    
    # 5. Microstructure-Enhanced Final Factor
    # Spread estimation
    df['daily_range'] = df['high'] - df['low']
    df['intraday_range'] = abs(df['close'] - df['open'])
    df['spread_ratio'] = df['intraday_range'] / (df['daily_range'] + 1e-8)
    
    # Order flow imbalance (price-volume correlation)
    df['price_volume_corr_5d'] = df['returns'].rolling(window=5).corr(df['volume_change'])
    
    # Final factor synthesis
    df['final_factor'] = (df['regime_alpha'] * 
                         (1 - 0.2 * df['spread_ratio']) * 
                         (1 + 0.3 * df['price_volume_corr_5d']))
    
    # Normalize the final factor
    factor_series = df['final_factor']
    factor_series = (factor_series - factor_series.rolling(window=20).mean()) / (factor_series.rolling(window=20).std() + 1e-8)
    
    return factor_series
