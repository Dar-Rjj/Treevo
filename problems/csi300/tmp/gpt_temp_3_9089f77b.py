import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    # Short-term volatility (5-day high-low range)
    data['short_term_vol'] = (data['high'] - data['low']).rolling(window=5).mean()
    
    # Medium-term volatility (20-day ATR efficiency)
    data['tr'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    data['medium_term_vol'] = data['tr'].rolling(window=20).mean()
    
    # Volatility ratio for regime classification
    data['vol_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    data['vol_ratio_change'] = data['vol_ratio'].pct_change()
    
    # Regime classification
    def classify_regime(row):
        if pd.isna(row['vol_ratio_change']):
            return 'transition'
        elif abs(row['vol_ratio_change']) > 0.15:
            return 'high'
        elif abs(row['vol_ratio_change']) < 0.05:
            return 'low'
        else:
            return 'transition'
    
    data['regime'] = data.apply(classify_regime, axis=1)
    
    # Multi-Scale Efficiency Divergence
    # Intraday efficiency
    data['intraday_eff'] = (abs(data['close'] - data['open']) / 
                           (data['high'] - data['low'] + 1e-8) * 
                           data['volume'] / (data['high'] - data['low'] + 1e-8))
    
    # Short-term efficiency (5-day)
    data['short_term_eff'] = ((data['close'] / data['close'].shift(4) - 1) * 
                             (data['volume'] / data['volume'].shift(4) - 1))
    
    # Medium-term efficiency (15-day)
    data['medium_term_eff'] = ((data['close'] / data['close'].shift(14) - 1) * 
                              (data['volume'] / data['volume'].shift(14) - 1))
    
    # Liquidity Acceleration & Stability
    # Amount efficiency
    data['amount_eff'] = abs(data['close'] - data['open']) / (data['amount'] + 1e-8)
    
    # Liquidity acceleration
    data['amount_ma_10'] = data['amount'].rolling(window=10).mean()
    data['liquidity_accel'] = (data['amount'] / data['amount_ma_10']) - 1
    
    # Volume stability
    data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
    data['volume_std_10'] = data['volume'].rolling(window=10).std()
    data['volume_stability'] = data['volume_ma_10'] / (data['volume_std_10'] + 1e-8)
    
    # Momentum Divergence Enhancement
    # Short-term momentum
    data['short_momentum'] = data['close'] / data['close'].shift(5) - 1
    
    # Medium-term momentum
    data['medium_momentum'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum divergence
    data['momentum_divergence'] = (data['close'] / data['close'].shift(5)) - (data['close'] / data['close'].shift(10))
    
    # Session-Based Concentration Features
    # Opening alignment
    data['opening_alignment'] = (np.sign(data['open'] - data['close'].shift(1)) * 
                                np.sign(data['close'] - data['open']))
    
    # Volume concentration (simplified - using daily patterns)
    data['volume_rank'] = data['volume'].rolling(window=20).apply(
        lambda x: (x[-1] - x.mean()) / (x.std() + 1e-8) if len(x) == 20 else np.nan
    )
    data['volume_concentration'] = np.where(data['volume_rank'] > 1, 1.2, 
                                           np.where(data['volume_rank'] < -1, 0.8, 1.0))
    
    # Regime-Adaptive Signal Construction
    def calculate_regime_signal(row):
        if row['regime'] == 'high':
            # High Volatility Regime
            core_div = row['intraday_eff'] * row['volume_concentration']
            momentum_enh = core_div * row['momentum_divergence']
            liquidity_inc = momentum_enh * abs(row['liquidity_accel'])
            return liquidity_inc * 1.2
            
        elif row['regime'] == 'low':
            # Low Volatility Regime
            core_div = row['medium_term_eff'] * row['volume_stability']
            momentum_inc = core_div * row['short_momentum']
            stability_enh = momentum_inc * row['amount_eff']
            return stability_enh * 0.9
            
        else:
            # Transition Regime
            multi_scale = row['intraday_eff'] * row['short_term_eff'] * row['medium_term_eff']
            momentum_align = multi_scale * row['momentum_divergence']
            liquidity_bal = momentum_align * row['volume_stability']
            session_align = liquidity_bal * row['opening_alignment']
            return session_align
    
    # Calculate final factor
    data['factor'] = data.apply(calculate_regime_signal, axis=1)
    
    # Clean up and return
    factor_series = data['factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor_series
