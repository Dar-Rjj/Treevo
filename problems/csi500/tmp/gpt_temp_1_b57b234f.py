import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Divergence with Volume Acceleration factor
    """
    # Multi-Timeframe Momentum Calculation
    df = df.copy()
    
    # Price and Volume Momentum for different timeframes
    for period in [5, 10, 20]:
        df[f'price_momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1
        df[f'volume_momentum_{period}d'] = df['volume'] / df['volume'].shift(period) - 1
    
    # Exponential Smoothing Application
    alpha = 0.3
    
    # EMA for price momentum across timeframes
    price_momentum_ema = pd.DataFrame()
    volume_momentum_ema = pd.DataFrame()
    
    for period in [5, 10, 20]:
        price_momentum_ema[f'ema_price_{period}d'] = df[f'price_momentum_{period}d'].ewm(alpha=alpha).mean()
        volume_momentum_ema[f'ema_volume_{period}d'] = df[f'volume_momentum_{period}d'].ewm(alpha=alpha).mean()
    
    # Calculate acceleration (first difference of EMA)
    price_acceleration = price_momentum_ema.diff()
    volume_acceleration = volume_momentum_ema.diff()
    
    # Volatility Regime Detection
    df['price_volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
    volatility_quantile = df['price_volatility_20d'].quantile(0.7)
    
    # Regime classification
    high_vol_regime = df['price_volatility_20d'] > volatility_quantile
    low_vol_regime = df['price_volatility_20d'] <= volatility_quantile
    
    # Divergence Pattern Analysis
    divergence_signals = pd.DataFrame(index=df.index)
    
    for period in [5, 10, 20]:
        # Directional divergence (sign mismatch)
        price_dir = np.sign(price_momentum_ema[f'ema_price_{period}d'])
        volume_dir = np.sign(volume_momentum_ema[f'ema_volume_{period}d'])
        divergence_signals[f'dir_div_{period}d'] = (price_dir != volume_dir).astype(int)
        
        # Magnitude divergence (relative strength)
        price_strength = price_momentum_ema[f'ema_price_{period}d'].abs()
        volume_strength = volume_momentum_ema[f'ema_volume_{period}d'].abs()
        divergence_signals[f'mag_div_{period}d'] = (price_strength - volume_strength) / (price_strength + volume_strength + 1e-8)
        
        # Acceleration divergence
        price_acc = price_acceleration[f'ema_price_{period}d']
        volume_acc = volume_acceleration[f'ema_volume_{period}d']
        divergence_signals[f'acc_div_{period}d'] = (price_acc - volume_acc) / (np.abs(price_acc) + np.abs(volume_acc) + 1e-8)
    
    # Multi-timeframe consistency
    short_term_consistency = (divergence_signals['dir_div_5d'] + 
                             divergence_signals['dir_div_10d']) / 2
    long_term_consistency = divergence_signals['dir_div_20d']
    
    # Factor Construction with Regime-Aware Weighting
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for idx in df.index:
        if high_vol_regime.loc[idx]:
            # High volatility: Volume emphasis (60% volume, 40% price)
            weight_price = 0.4
            weight_volume = 0.6
        elif low_vol_regime.loc[idx]:
            # Low volatility: Price emphasis (70% price, 30% volume)
            weight_price = 0.7
            weight_volume = 0.3
        else:
            # Mixed regime: Balanced approach
            weight_price = 0.5
            weight_volume = 0.5
        
        # Combine signals with regime weights
        price_signal = (divergence_signals.loc[idx, 'mag_div_5d'] * 0.4 + 
                       divergence_signals.loc[idx, 'mag_div_10d'] * 0.35 + 
                       divergence_signals.loc[idx, 'mag_div_20d'] * 0.25)
        
        volume_signal = (divergence_signals.loc[idx, 'acc_div_5d'] * 0.4 + 
                        divergence_signals.loc[idx, 'acc_div_10d'] * 0.35 + 
                        divergence_signals.loc[idx, 'acc_div_20d'] * 0.25)
        
        # Consistency adjustment
        consistency_score = (short_term_consistency.loc[idx] * 0.6 + 
                           long_term_consistency.loc[idx] * 0.4)
        
        # Final factor value
        base_factor = (price_signal * weight_price + volume_signal * weight_volume)
        factor_values.loc[idx] = base_factor * (1 + 0.2 * consistency_score)
    
    # Volatility Scaling
    recent_vol = df['price_volatility_20d'].rolling(window=5).mean()
    factor_values = factor_values / (recent_vol + 1e-8)
    
    # Signal classification
    signal_threshold = factor_values.quantile(0.6)
    bearish_threshold = factor_values.quantile(0.4)
    
    # Final alpha output with signal interpretation
    alpha_signal = pd.Series(index=df.index, dtype=float)
    for idx in df.index:
        if factor_values.loc[idx] > signal_threshold:
            # Bullish confirmation: Positive divergence with volume acceleration
            alpha_signal.loc[idx] = factor_values.loc[idx] * 1.2
        elif factor_values.loc[idx] < bearish_threshold:
            # Bearish confirmation: Negative divergence
            alpha_signal.loc[idx] = factor_values.loc[idx] * 0.8
        else:
            # Neutral/Uncertain signal
            alpha_signal.loc[idx] = factor_values.loc[idx] * 1.0
    
    return alpha_signal
