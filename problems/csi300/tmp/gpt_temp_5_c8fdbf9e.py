import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Momentum Divergence
    short_term_momentum = (df['high'] - df['open']) / (df['open'] - df['low']).replace(0, np.nan)
    short_term_avg = short_term_momentum.rolling(window=3, min_periods=1).mean()
    
    long_term_reversal = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    long_term_avg = long_term_reversal.rolling(window=10, min_periods=1).mean()
    
    volume_trend = df['volume'].rolling(window=5, min_periods=1).mean() / df['volume'].rolling(window=20, min_periods=1).mean()
    divergence_factor = (short_term_avg - long_term_avg) * volume_trend
    
    # Volatility-Adjusted Price-Volume Correlation
    price_volume_corr = df['close'].rolling(window=10).corr(df['volume'])
    volatility_regime = (df['high'] - df['low']).rolling(window=20, min_periods=1).mean() / (df['high'] - df['low']).rolling(window=5, min_periods=1).mean()
    recent_return_sign = np.sign(df['close'].pct_change(periods=2))
    vol_adjusted_signal = price_volume_corr * volatility_regime * recent_return_sign
    
    # Amplitude-Decay Oscillator
    price_amplitude = (df['high'] - df['low']) / df['close']
    amplitude_ewm = price_amplitude.ewm(alpha=0.1, adjust=False).mean()
    amplitude_decay = amplitude_ewm.rolling(window=3, min_periods=1).mean() - amplitude_ewm.rolling(window=10, min_periods=1).mean()
    volume_acceleration = df['volume'] / df['volume'].shift(1)
    amplitude_signal = amplitude_decay * volume_acceleration
    
    # Calculate consecutive up/down days
    returns = df['close'].pct_change()
    consecutive_days = returns.rolling(window=5).apply(lambda x: len([i for i in range(1, len(x)) if np.sign(x[i]) == np.sign(x[i-1])]), raw=True)
    amplitude_oscillator = amplitude_signal * consecutive_days
    
    # Liquidity-Efficient Price Movement
    price_efficiency = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    price_efficiency_ma = price_efficiency.rolling(window=5, min_periods=1).mean()
    liquidity_impact = (df['volume'] / df['amount']).replace([np.inf, -np.inf], np.nan).rolling(window=3, min_periods=1).mean()
    combined_signal = price_efficiency_ma * liquidity_impact
    recent_volatility = (df['high'] - df['low']).rolling(window=5, min_periods=1).mean()
    liquidity_signal = combined_signal / recent_volatility.replace(0, np.nan)
    
    # Regime-Switching Mean Reversion
    trend_strength = df['close'].pct_change(periods=10).abs()
    volatility_threshold = (df['high'] - df['low']).rolling(window=20, min_periods=1).std()
    high_vol_regime = trend_strength > volatility_threshold
    
    ma_5 = df['close'].rolling(window=5, min_periods=1).mean()
    ma_10 = df['close'].rolling(window=10, min_periods=1).mean()
    price_range_20 = (df['high'] - df['low']).rolling(window=20, min_periods=1).mean()
    
    mean_reversion_high_vol = (df['close'] - ma_5) / (df['high'] - df['low']).replace(0, np.nan)
    mean_reversion_low_vol = (df['close'] - ma_10) / price_range_20.replace(0, np.nan)
    
    conditional_mean_reversion = np.where(high_vol_regime, mean_reversion_high_vol, mean_reversion_low_vol)
    
    volume_percentile = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    regime_signal = conditional_mean_reversion * volume_percentile
    
    # Combine all factors with equal weights
    final_factor = (
        divergence_factor.fillna(0) + 
        vol_adjusted_signal.fillna(0) + 
        amplitude_oscillator.fillna(0) + 
        liquidity_signal.fillna(0) + 
        regime_signal.fillna(0)
    )
    
    return final_factor
