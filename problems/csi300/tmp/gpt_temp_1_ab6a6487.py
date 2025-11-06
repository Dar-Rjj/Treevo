import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Liquidity-Momentum Synthesis factor combining gap dynamics, 
    volatility transmission, liquidity efficiency, and trade distribution signals.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need sufficient history for calculations
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        prev_close = df.iloc[i-1]['close'] if i > 0 else current['open']
        
        # Opening Gap Dynamics
        gap_amplitude = abs(current['open'] / prev_close - 1) if prev_close != 0 else 0
        
        # Intraday Reversal (using current day's range)
        day_range = current['high'] - current['low']
        intraday_reversal = (current['close'] - current['open']) / day_range if day_range != 0 else 0
        
        # Opening Volume Intensity (approximated using first hour vs total volume)
        # Since we don't have intraday breakdown, we'll use opening volume pattern
        volume_intensity = current['volume'] / df.iloc[max(0, i-5):i+1]['volume'].mean() if i >= 5 else 1
        
        # Volatility Transmission
        recent_ranges = [df.iloc[j]['high'] - df.iloc[j]['low'] for j in range(max(0, i-19), i+1)]
        avg_range_20d = np.mean(recent_ranges) if recent_ranges else day_range
        volatility_state = day_range / avg_range_20d if avg_range_20d != 0 else 1
        
        # Sector Correlation (approximated using stock's own momentum vs market)
        stock_returns = [df.iloc[j]['close'] / df.iloc[j-1]['close'] - 1 for j in range(max(1, i-9), i+1) if j > 0]
        market_returns = [(df.iloc[j]['high'] - df.iloc[j]['low']) / df.iloc[j]['open'] for j in range(max(0, i-9), i+1)]
        
        if len(stock_returns) > 1 and len(market_returns) > 1:
            sector_correlation = np.corrcoef(stock_returns[:min(10, len(stock_returns))], 
                                           market_returns[:min(10, len(market_returns))])[0,1]
            sector_correlation = 0 if np.isnan(sector_correlation) else sector_correlation
        else:
            sector_correlation = 0
        
        # Liquidity Efficiency
        range_efficiency = abs(current['close'] - prev_close) / day_range if day_range != 0 else 0
        
        price_change = abs(current['close'] - prev_close)
        price_impact = current['volume'] / price_change if price_change != 0 else current['volume']
        
        # Momentum Quality (5-day momentum)
        close_5d_ago = df.iloc[i-5]['close'] if i >= 5 else prev_close
        momentum_quality = current['close'] / close_5d_ago - 1 if close_5d_ago != 0 else 0
        
        # Trade Distribution (approximated using volume patterns)
        recent_volumes = [df.iloc[j]['volume'] for j in range(max(0, i-9), i+1)]
        avg_volume_10d = np.mean(recent_volumes) if recent_volumes else current['volume']
        
        # Block dominance (large volume days)
        block_dominance = current['volume'] / avg_volume_10d if avg_volume_10d != 0 else 1
        
        # Retail participation (small volume days inverse)
        retail_participation = 1 / block_dominance if block_dominance != 0 else 1
        
        # Composite Signals
        gap_reversal = gap_amplitude * intraday_reversal * volume_intensity
        volatility_momentum = momentum_quality * volatility_state * (1 + sector_correlation)
        liquidity_trend = range_efficiency * (1 / price_impact if price_impact != 0 else 0) * momentum_quality
        microstructure_momentum = block_dominance * retail_participation * momentum_quality
        
        # Final composite factor
        factor_value = (
            gap_reversal * 0.25 +
            volatility_momentum * 0.30 +
            liquidity_trend * 0.25 +
            microstructure_momentum * 0.20
        )
        
        result.iloc[i] = factor_value
    
    return result
