import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Microstructure Liquidity Breakout Factor
    """
    data = df.copy()
    
    # Dual Volatility Assessment
    data['ret_20d_vol'] = data['close'].pct_change().rolling(window=20).std()
    
    # True Range Volatility
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['tr_vol'] = data['true_range'].rolling(window=20).std()
    
    # Microstructure Noise Ratio
    data['overnight_gap'] = abs(data['open'] - data['prev_close'])
    data['intraday_vol'] = (data['high'] - data['low']).rolling(window=5).std()
    data['noise_ratio'] = data['intraday_vol'] / (data['overnight_gap'].replace(0, np.nan).rolling(window=5).mean())
    
    # Regime Classification with Liquidity Context
    data['vol_vs_median'] = data['ret_20d_vol'] / data['ret_20d_vol'].rolling(window=60).median()
    
    # Volume Clustering Persistence
    data['vol_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['vol_above_avg'] = (data['volume'] > data['vol_20d_avg']).astype(int)
    data['vol_cluster_count'] = data['vol_above_avg'].rolling(window=10, min_periods=1).apply(
        lambda x: max((x == 1).cumsum() - (x == 1).cumsum().where(x == 0).ffill().fillna(0)), 
        raw=False
    )
    
    # Microstructure Regime Classification
    data['high_noise_regime'] = ((data['vol_vs_median'] > 1.2) & 
                                (data['noise_ratio'] > data['noise_ratio'].rolling(window=60).median())).astype(int)
    
    # Multi-timeframe Breakout Strength
    data['dist_from_50d_high'] = (data['close'] - data['high'].rolling(window=50).max()) / data['close']
    data['dist_from_50d_low'] = (data['close'] - data['low'].rolling(window=50).min()) / data['close']
    data['closing_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Liquidity Confirmation
    data['volume_price_corr'] = data['volume'].rolling(window=10).corr(data['close'])
    
    # Momentum-Liquidity-Pressure Convergence
    data['mom_5d'] = data['close'].pct_change(5)
    data['mom_10d'] = data['close'].pct_change(10)
    data['mom_20d'] = data['close'].pct_change(20)
    data['mom_alignment'] = ((data['mom_5d'] > 0) & (data['mom_10d'] > 0) & (data['mom_20d'] > 0)).astype(int) - \
                           ((data['mom_5d'] < 0) & (data['mom_10d'] < 0) & (data['mom_20d'] < 0)).astype(int)
    
    data['vol_mom_5d'] = data['volume'].pct_change(5)
    data['vol_mom_10d'] = data['volume'].pct_change(10)
    data['vol_mom_20d'] = data['volume'].pct_change(20)
    
    # Cumulative Net Pressure
    data['net_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['cum_net_pressure_3d'] = data['net_pressure'].rolling(window=3).sum()
    data['cum_net_pressure_5d'] = data['net_pressure'].rolling(window=5).sum()
    
    # Volume-Liquidity Surge Validation
    data['volume_spike'] = (data['volume'] > 2 * data['vol_20d_avg']).astype(int)
    data['liquidity_absorption'] = (data['close'] - data['open']) / data['volume'].rolling(window=5).sum().replace(0, np.nan)
    
    # Intraday Efficiency with Liquidity Patterns
    data['overnight_gap_resolution'] = abs(data['open'] - data['prev_close']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Active Liquidity Efficiency
    data['active_liquidity_eff'] = ((data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)) * data['volume']
    
    # Micro-Reversal Detection
    data['micro_reversal'] = abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Reversal Risk with Liquidity Signals
    data['mom_compression'] = abs(data['mom_5d']) / (abs(data['mom_10d']).replace(0, np.nan))
    
    # Multi-Timeframe Efficiency Divergence
    data['price_range_efficiency_trend'] = ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)).rolling(window=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan, raw=False
    )
    
    # Adaptive Microstructure-Liquidity Integration
    # Core Breakout Momentum
    data['core_breakout'] = (data['dist_from_50d_high'] * 0.4 + 
                            data['closing_efficiency'] * 0.3 + 
                            data['mom_alignment'] * 0.3)
    
    # Regime Multiplier
    data['regime_multiplier'] = np.where(
        data['high_noise_regime'] == 1,
        # High Volatility/High-Noise regime
        (1 + data['volume_spike'] * 0.2) * (1 + data['vol_cluster_count'] * 0.1),
        # Low Volatility/Low-Noise regime  
        (1 + data['volume_price_corr'] * 0.3) * (1 - data['micro_reversal'] * 0.2)
    )
    
    # Volume-Liquidity Surge Filter
    data['volume_liquidity_filter'] = np.where(
        data['volume_spike'] == 1,
        data['liquidity_absorption'] * data['active_liquidity_eff'].rolling(window=5).mean(),
        1.0
    )
    
    # Microstructure Efficiency Adjustment
    data['microstructure_adj'] = 1 - (data['overnight_gap_resolution'] * 0.2 + 
                                     data['micro_reversal'] * 0.3 + 
                                     data['mom_compression'] * 0.5)
    
    # Reversal Risk with Liquidity Divergence Adjustment
    data['reversal_risk_adj'] = np.where(
        (data['mom_compression'] > 1.5) & (data['volume_price_corr'] < 0),
        0.7,  # High reversal risk
        np.where(data['mom_compression'] > 1.2, 0.9, 1.0)  # Moderate to low risk
    )
    
    # Final Factor Construction
    factor = (data['core_breakout'] * 
             data['regime_multiplier'] * 
             data['volume_liquidity_filter'] * 
             data['microstructure_adj'] * 
             data['reversal_risk_adj'])
    
    return factor
