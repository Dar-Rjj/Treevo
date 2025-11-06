import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining intraday pressure cumulation with volatility context,
    liquidity-clustered range expansion, efficiency-based order flow divergence,
    volatility-regime adaptive reversal detection, gap absorption dynamics,
    and price-volume synchronization with trend context.
    """
    data = df.copy()
    
    # Intraday Pressure Cumulation with Volatility Context
    # Calculate Asymmetric Pressure Components
    data['Buy_Pressure'] = np.where(data['close'] > data['open'], data['close'] - data['low'], 0)
    data['Sell_Pressure'] = np.where(data['close'] < data['open'], data['high'] - data['close'], 0)
    data['Net_Pressure'] = data['Buy_Pressure'] - data['Sell_Pressure']
    
    # Cumulative Pressure with Volume Confirmation
    data['Cum_Pressure_3d'] = data['Net_Pressure'].rolling(window=3, min_periods=1).sum()
    data['Volume_5d_Avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['Volume_Ratio'] = data['volume'] / data['Volume_5d_Avg']
    data['Weighted_Pressure'] = data['Cum_Pressure_3d'] * data['Volume_Ratio']
    
    # Volatility-Scaled Momentum Adjustment
    data['Prev_Close'] = data['close'].shift(1)
    data['TR1'] = data['high'] - data['low']
    data['TR2'] = abs(data['high'] - data['Prev_Close'])
    data['TR3'] = abs(data['low'] - data['Prev_Close'])
    data['True_Range'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    data['Momentum_5d'] = data['close'] - data['close'].shift(5)
    data['Momentum_Scaled'] = abs(data['Momentum_5d']) / data['True_Range']
    data['Pressure_Volatility_Scaled'] = data['Weighted_Pressure'] * data['Momentum_Scaled'] * np.sign(data['Weighted_Pressure'])
    
    # Liquidity-Clustered Range Expansion Signals
    # Detect Range Expansion Events
    data['Daily_Range'] = (data['high'] - data['low']) / data['open']
    data['Range_5d_Avg'] = data['Daily_Range'].rolling(window=5, min_periods=1).mean()
    data['Range_Expansion'] = data['Daily_Range'] > (1.5 * data['Range_5d_Avg'])
    data['Expansion_Magnitude'] = (data['Daily_Range'] - data['Range_5d_Avg']) / data['Range_5d_Avg']
    
    # Identify Liquidity Clusters
    data['Liquidity'] = data['amount'] / data['volume']
    data['Liquidity_10d_Avg'] = data['Liquidity'].rolling(window=10, min_periods=1).mean()
    data['Liquidity_Cluster'] = data['Liquidity'] > (2 * data['Liquidity_10d_Avg'])
    data['Cluster_Magnitude'] = (data['Liquidity'] - data['Liquidity_10d_Avg']) / data['Liquidity_10d_Avg']
    
    # Combined Expansion-Liquidity Signal
    data['Range_Position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['Expansion_Liquidity_Signal'] = data['Range_Position'] * data['Cluster_Magnitude'] * data['Volume_Ratio']
    data['Expansion_Persistence'] = data['Range_Expansion'].rolling(window=3, min_periods=1).sum()
    data['Expansion_Liquidity_Signal'] *= data['Expansion_Persistence']
    
    # Efficiency-Based Order Flow Divergence
    # Compute Price Efficiency
    data['Intraday_Efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['Efficiency_Abs'] = abs(data['Intraday_Efficiency'])
    
    # Calculate Order Flow Imbalance
    data['Typical_Price'] = (data['high'] + data['low'] + data['close']) / 3
    data['Price_Change'] = data['Typical_Price'].diff()
    data['Volume_Change'] = data['volume'].diff()
    
    # 3-day cumulative price-volume correlation
    corr_window = 3
    price_volume_corr = []
    for i in range(len(data)):
        if i < corr_window - 1:
            price_volume_corr.append(0)
        else:
            window_data = data.iloc[i-corr_window+1:i+1]
            if window_data['Price_Change'].std() > 0 and window_data['Volume_Change'].std() > 0:
                corr = window_data['Price_Change'].corr(window_data['Volume_Change'])
                price_volume_corr.append(corr if not np.isnan(corr) else 0)
            else:
                price_volume_corr.append(0)
    
    data['Price_Volume_Corr_3d'] = price_volume_corr
    data['Flow_Divergence'] = data['Efficiency_Abs'] - abs(data['Price_Volume_Corr_3d'])
    
    # Volume-Confirmed Divergence Momentum
    data['Volume_10d_Avg'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['Volume_Ratio_10d'] = data['volume'] / data['Volume_10d_Avg']
    data['Divergence_Momentum'] = data['Flow_Divergence'] - data['Flow_Divergence'].shift(2)
    data['Direction_Persistence'] = np.sign(data['Flow_Divergence']).rolling(window=3, min_periods=1).sum()
    data['Divergence_Signal'] = data['Divergence_Momentum'] * data['Volume_Ratio_10d'] * data['Direction_Persistence']
    
    # Volatility-Regime Adaptive Reversal Detection
    # Classify Volatility Regimes
    data['Range_10d'] = (data['high'] - data['low']) / data['open']
    data['Range_20d_Pct'] = data['Range_10d'].rolling(window=20, min_periods=1).rank(pct=True)
    data['High_Vol_Regime'] = data['Range_20d_Pct'] > 0.7
    data['Low_Vol_Regime'] = data['Range_20d_Pct'] < 0.3
    
    # Liquidity-Based Reversal Patterns
    data['Intraday_Return'] = (data['close'] - data['open']) / data['open']
    data['Reversal_Strength'] = -data['Intraday_Return'] * data['Cluster_Magnitude']
    
    # Regime-Adaptive Signal Generation
    data['High_Vol_Signal'] = data['Reversal_Strength'] * data['Expansion_Magnitude'] * data['High_Vol_Regime']
    data['Low_Vol_Signal'] = data['Divergence_Signal'] * data['Low_Vol_Regime']
    data['Regime_Confidence'] = abs(data['Range_20d_Pct'] - 0.5) * 2
    data['Regime_Adaptive_Signal'] = (data['High_Vol_Signal'] + data['Low_Vol_Signal']) * data['Regime_Confidence']
    
    # Gap Absorption with Pressure Cumulation
    # Measure Gap Absorption Dynamics
    data['Gap_Magnitude'] = abs(data['open'] - data['Prev_Close']) / data['Prev_Close']
    data['Potential_Absorption'] = (data['high'] - data['open']) / abs(data['open'] - data['Prev_Close'])
    data['Actual_Absorption'] = (data['close'] - data['open']) / abs(data['open'] - data['Prev_Close'])
    data['Absorption_Ratio'] = data['Actual_Absorption'] / data['Potential_Absorption']
    data['Absorption_Ratio'] = data['Absorption_Ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Combine with Intraday Pressure
    data['Absorption_Pressure_Signal'] = data['Absorption_Ratio'] * data['Net_Pressure'] * data['Volume_Ratio']
    data['Absorption_Alignment'] = (np.sign(data['Absorption_Ratio']) == np.sign(data['Net_Pressure'])).astype(int)
    data['Absorption_Persistence'] = data['Absorption_Alignment'].rolling(window=3, min_periods=1).sum()
    data['Gap_Absorption_Signal'] = data['Absorption_Pressure_Signal'] * data['Gap_Magnitude'] * data['Absorption_Persistence']
    
    # Price-Volume Synchronization with Trend Context
    # Compute Intraday Synchronization
    daily_corr = []
    for i in range(len(data)):
        if i == 0:
            daily_corr.append(0)
        else:
            intraday_prices = [data['open'].iloc[i], data['high'].iloc[i], data['low'].iloc[i], data['close'].iloc[i]]
            intraday_volumes = [data['volume'].iloc[i]] * 4
            if np.std(intraday_prices) > 0 and np.std(intraday_volumes) > 0:
                corr = np.corrcoef(intraday_prices, intraday_volumes)[0, 1]
                daily_corr.append(corr if not np.isnan(corr) else 0)
            else:
                daily_corr.append(0)
    
    data['Intraday_Corr'] = daily_corr
    data['Sync_Score'] = 1 - abs(data['Intraday_Corr'])
    
    # Trend Consistency Measurement
    data['MA_5'] = data['close'].rolling(window=5, min_periods=1).mean()
    data['MA_10'] = data['close'].rolling(window=10, min_periods=1).mean()
    data['Trend_Alignment'] = ((data['close'] > data['MA_5']) & (data['MA_5'] > data['MA_10'])).astype(int) - \
                             ((data['close'] < data['MA_5']) & (data['MA_5'] < data['MA_10'])).astype(int)
    
    # Synchronized Trend Signals
    data['Sync_Trend_Signal'] = data['Sync_Score'] * data['Trend_Alignment'] * data['Volume_Ratio']
    data['Sync_Trend_Cumulative'] = data['Sync_Trend_Signal'].rolling(window=3, min_periods=1).sum()
    data['Sync_Trend_Final'] = data['Sync_Trend_Cumulative'] * (1 / (1 + data['True_Range']))
    
    # Combine all signals with appropriate weights
    weights = {
        'pressure': 0.25,
        'expansion': 0.20,
        'divergence': 0.15,
        'regime': 0.15,
        'gap': 0.15,
        'sync': 0.10
    }
    
    # Normalize signals before combination
    signals_to_combine = [
        data['Pressure_Volatility_Scaled'],
        data['Expansion_Liquidity_Signal'],
        data['Divergence_Signal'],
        data['Regime_Adaptive_Signal'],
        data['Gap_Absorption_Signal'],
        data['Sync_Trend_Final']
    ]
    
    # Z-score normalization for each signal
    normalized_signals = []
    for signal in signals_to_combine:
        mean_val = signal.rolling(window=20, min_periods=1).mean()
        std_val = signal.rolling(window=20, min_periods=1).std()
        normalized = (signal - mean_val) / std_val
        normalized_signals.append(normalized.replace([np.inf, -np.inf], 0).fillna(0))
    
    # Weighted combination
    final_signal = (
        weights['pressure'] * normalized_signals[0] +
        weights['expansion'] * normalized_signals[1] +
        weights['divergence'] * normalized_signals[2] +
        weights['regime'] * normalized_signals[3] +
        weights['gap'] * normalized_signals[4] +
        weights['sync'] * normalized_signals[5]
    )
    
    return final_signal
