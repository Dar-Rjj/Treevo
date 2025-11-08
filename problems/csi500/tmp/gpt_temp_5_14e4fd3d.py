import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required technical indicators
    data['prev_close'] = data['close'].shift(1)
    data['range'] = data['high'] - data['low']
    data['avg_range_20'] = data['range'].rolling(window=20).mean()
    data['avg_volume_3'] = data['volume'].rolling(window=3).mean()
    data['avg_volume_5'] = data['volume'].rolling(window=5).mean()
    
    # ATR calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5'] = data['tr'].rolling(window=5).mean()
    data['atr_10'] = data['tr'].rolling(window=10).mean()
    data['atr_20'] = data['tr'].rolling(window=20).mean()
    
    # Rolling highs and lows
    data['high_20'] = data['high'].rolling(window=20).max()
    data['low_20'] = data['low'].rolling(window=20).min()
    
    # Returns
    data['return_3d'] = data['close'].pct_change(3)
    data['return_5d'] = data['close'].pct_change(5)
    
    # Sector volume proxy (using rolling average as sector proxy)
    data['sector_avg_volume'] = data['volume'].rolling(window=20).mean()
    
    for i in range(len(data)):
        if i < 20:  # Need sufficient history
            result.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        
        # 1. Range Compression Breakout with Volume Confirmation
        if current['avg_range_20'] > 0:
            range_compression = current['range'] / current['avg_range_20']
            cross_asset_volume = current['volume'] / current['sector_avg_volume']
            signal1 = range_compression * cross_asset_volume
        else:
            signal1 = 0
            
        # 2. Gap Absorption with Momentum Divergence and Order Flow
        if i > 0 and abs(data.iloc[i-1]['close'] - current['open']) > 0:
            gap_absorption = (current['close'] - current['open']) / abs(current['open'] - data.iloc[i-1]['close'])
            momentum_decay = current['return_3d'] - current['return_5d'] if not pd.isna(current['return_3d']) and not pd.isna(current['return_5d']) else 0
            
            # Order flow imbalance proxy using amount/volume as price pressure indicator
            if current['volume'] > 0:
                avg_price = current['amount'] / current['volume'] if current['amount'] > 0 else current['close']
                price_pressure = (current['close'] - avg_price) / avg_price
                order_flow_imbalance = price_pressure
            else:
                order_flow_imbalance = 0
                
            signal2 = gap_absorption * momentum_decay * order_flow_imbalance
        else:
            signal2 = 0
            
        # 3. Intraday Pressure with Volume Acceleration and Liquidity
        if current['high'] != current['low']:
            daily_pressure = ((current['close'] - current['low']) - (current['high'] - current['close'])) / (current['high'] - current['low'])
            
            if current['avg_volume_3'] > 0:
                volume_acceleration = current['volume'] / current['avg_volume_3']
            else:
                volume_acceleration = 1
                
            # Liquidity gradient proxy using volatility-based measure
            if current['atr_5'] > 0:
                liquidity_gradient = (current['atr_5'] - current['atr_20']) / current['atr_20'] if current['atr_20'] > 0 else 0
            else:
                liquidity_gradient = 0
                
            signal3 = daily_pressure * volume_acceleration * liquidity_gradient
        else:
            signal3 = 0
            
        # 4. Volatility-Regime Breakout with Structural Confirmation
        if current['atr_20'] > 0:
            volatility_compression = current['atr_5'] / current['atr_20']
        else:
            volatility_compression = 1
            
        range_breakout = 1 if (current['close'] > current['high_20'] or current['close'] < current['low_20']) else 0
        
        if current['avg_volume_5'] > 0:
            volume_confirmation = current['volume'] / current['avg_volume_5']
        else:
            volume_confirmation = 1
            
        signal4 = volatility_compression * range_breakout * volume_confirmation
        
        # 5. Amplitude-Adjusted Gap with Microstructure Persistence
        if i > 0 and current['atr_10'] > 0:
            gap_size = abs(current['open'] - data.iloc[i-1]['close']) / current['atr_10']
            relative_amplitude = current['range'] / current['close'] if current['close'] > 0 else 0
            
            # Market maker positioning proxy using volume volatility
            if i >= 10:
                volume_volatility = data['volume'].iloc[i-9:i+1].std() / data['volume'].iloc[i-9:i+1].mean() if data['volume'].iloc[i-9:i+1].mean() > 0 else 0
                market_maker_positioning = 1 / (1 + volume_volatility)  # Inverse relationship
            else:
                market_maker_positioning = 1
                
            if relative_amplitude > 0:
                signal5 = (gap_size / relative_amplitude) * market_maker_positioning
            else:
                signal5 = 0
        else:
            signal5 = 0
            
        # 6. Multi-Timeframe Pressure with Cross-Asset Momentum
        if current['high'] != current['low']:
            short_term_pressure = ((current['close'] - current['low']) - (current['high'] - current['close'])) / (current['high'] - current['low'])
        else:
            short_term_pressure = 0
            
        # Medium-term persistence (count consecutive same-sign pressure days)
        persistence_count = 1
        for j in range(1, min(6, i+1)):
            if data.iloc[i-j]['high'] != data.iloc[i-j]['low']:
                prev_pressure = ((data.iloc[i-j]['close'] - data.iloc[i-j]['low']) - (data.iloc[i-j]['high'] - data.iloc[i-j]['close'])) / (data.iloc[i-j]['high'] - data.iloc[i-j]['low'])
                if (prev_pressure > 0 and short_term_pressure > 0) or (prev_pressure < 0 and short_term_pressure < 0):
                    persistence_count += 1
                else:
                    break
            else:
                break
                
        # Cross-asset momentum proxy using relative performance vs rolling mean
        if not pd.isna(current['return_3d']):
            cross_asset_momentum = current['return_3d'] - data['return_3d'].rolling(window=20).mean().iloc[i] if not pd.isna(data['return_3d'].rolling(window=20).mean().iloc[i]) else current['return_3d']
        else:
            cross_asset_momentum = 0
            
        signal6 = short_term_pressure * persistence_count * cross_asset_momentum
        
        # Combine all signals with equal weighting
        combined_signal = (signal1 + signal2 + signal3 + signal4 + signal5 + signal6) / 6
        result.iloc[i] = combined_signal
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
