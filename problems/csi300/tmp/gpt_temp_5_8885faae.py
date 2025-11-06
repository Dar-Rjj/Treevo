import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Acceleration with Smart Money Confirmation factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate True Range for volatility regime detection
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility regime classification using rolling window
    window_size = 20
    df['tr_rank'] = df['true_range'].rolling(window=window_size).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Define volatility regimes
    df['vol_regime'] = 'normal'
    df.loc[df['tr_rank'] >= 0.7, 'vol_regime'] = 'high'
    df.loc[df['tr_rank'] <= 0.3, 'vol_regime'] = 'low'
    
    # Momentum Acceleration Analysis
    df['return_5d'] = df['close'].pct_change(5)
    df['acceleration'] = df['return_5d'].diff(3)  # 3-day change in 5-day returns
    
    # Range Efficiency
    df['daily_range'] = df['high'] - df['low']
    df['range_efficiency'] = abs(df['close'] - df['open']) / (df['daily_range'] + 1e-8)
    
    # Combine acceleration with range efficiency
    df['accel_efficiency'] = df['acceleration'] * df['range_efficiency']
    
    # Smart Money Flow Analysis
    df['avg_trade_size'] = df['amount'] / (df['volume'] + 1e-8)
    df['large_trade_threshold'] = df['avg_trade_size'].rolling(window=10).quantile(0.7)
    df['is_large_trade'] = df['avg_trade_size'] > df['large_trade_threshold']
    
    # Calculate net flow direction using large trades
    df['price_momentum'] = (df['close'] - df['open']) / df['open']
    df['smart_money_flow'] = df['is_large_trade'] * df['price_momentum']
    df['smart_money_flow_ma'] = df['smart_money_flow'].rolling(window=5).mean()
    
    # Gap Analysis
    df['opening_gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['gap_flow_combo'] = df['opening_gap'] * df['smart_money_flow_ma']
    
    # Pressure Accumulation Synthesis
    df['close_position'] = (df['close'] - df['low']) / (df['daily_range'] + 1e-8)
    df['volume_pressure'] = df['close_position'] * df['volume']
    df['pressure_accumulation'] = df['volume_pressure'].ewm(span=5).mean()
    df['pressure_trend'] = df['pressure_accumulation'].diff(3)
    
    # Regime-Adaptive Signal Generation
    for i in range(window_size, len(df)):
        current_data = df.iloc[i]
        regime = current_data['vol_regime']
        
        if regime == 'high':
            # Acceleration-Pressure Divergence in high volatility
            accel_signal = -df['accel_efficiency'].iloc[i]  # Focus on reversals
            pressure_signal = df['pressure_trend'].iloc[i]
            smart_money_signal = df['gap_flow_combo'].iloc[i]
            
            # Amplify divergence signals
            signal = (accel_signal * 0.6 + pressure_signal * 0.3 + 
                     smart_money_signal * 0.1)
            
        elif regime == 'low':
            # Acceleration Continuation in low volatility
            accel_signal = df['accel_efficiency'].iloc[i]
            pressure_signal = df['pressure_accumulation'].iloc[i]
            smart_money_signal = df['smart_money_flow_ma'].iloc[i]
            
            # Require alignment and use for timing
            alignment_factor = 1.0 if (accel_signal * smart_money_signal > 0) else 0.3
            signal = (accel_signal * 0.5 + pressure_signal * 0.3 + 
                     smart_money_signal * 0.2) * alignment_factor
            
        else:  # normal volatility
            # Balanced synthesis
            accel_signal = df['accel_efficiency'].iloc[i]
            pressure_signal = df['pressure_accumulation'].iloc[i]
            smart_money_signal = df['smart_money_flow_ma'].iloc[i]
            range_signal = df['range_efficiency'].iloc[i]
            
            # Equal weighting with confirmation filters
            base_signal = (accel_signal * 0.4 + pressure_signal * 0.4 + 
                          smart_money_signal * 0.2)
            
            # Use range efficiency for signal strength
            signal = base_signal * range_signal
        
        result.iloc[i] = signal
    
    # Clean up and return
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
