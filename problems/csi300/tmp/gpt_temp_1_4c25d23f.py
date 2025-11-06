import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence with Intraday Momentum Persistence
    """
    data = df.copy()
    
    # Calculate Multi-Timeframe Price-Volume Divergence
    # Short-Term Divergence (5-day)
    data['price_ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_ret_5d'] = data['volume'] / data['volume_ma_5d'] - 1
    data['divergence_5d'] = data['price_ret_5d'] - data['volume_ret_5d']
    
    # Medium-Term Divergence (21-day)
    data['price_ret_21d'] = data['close'] / data['close'].shift(21) - 1
    data['volume_ma_21d'] = data['volume'].rolling(window=21).mean()
    data['volume_ret_21d'] = data['volume'] / data['volume_ma_21d'] - 1
    data['divergence_21d'] = data['price_ret_21d'] - data['volume_ret_21d']
    
    # Ultra-Short Divergence (2-day)
    data['price_ret_2d'] = data['close'] / data['close'].shift(2) - 1
    data['volume_ma_2d'] = data['volume'].rolling(window=2).mean()
    data['volume_ret_2d'] = data['volume'] / data['volume_ma_2d'] - 1
    data['divergence_2d'] = data['price_ret_2d'] - data['volume_ret_2d']
    
    # Divergence Persistence Analysis
    for timeframe in ['2d', '5d', '21d']:
        col = f'divergence_{timeframe}'
        # Direction consistency
        data[f'{col}_sign'] = np.sign(data[col])
        data[f'{col}_persistence'] = data[f'{col}_sign'].groupby(
            (data[f'{col}_sign'] != data[f'{col}_sign'].shift(1)).cumsum()
        ).cumcount() + 1
        
        # Magnitude changes
        data[f'{col}_mag_change'] = data[col].diff()
    
    # Multi-timeframe alignment
    data['alignment_score'] = (
        (data['divergence_2d_sign'] == data['divergence_5d_sign']).astype(int) +
        (data['divergence_2d_sign'] == data['divergence_21d_sign']).astype(int) +
        (data['divergence_5d_sign'] == data['divergence_21d_sign']).astype(int)
    ) / 3.0
    
    # Combined divergence signal
    data['combined_divergence'] = (
        data['divergence_2d'] * data['divergence_2d_persistence'] +
        data['divergence_5d'] * data['divergence_5d_persistence'] +
        data['divergence_21d'] * data['divergence_21d_persistence']
    ) / (data['divergence_2d_persistence'] + data['divergence_5d_persistence'] + data['divergence_21d_persistence'])
    
    data['weighted_divergence'] = data['combined_divergence'] * data['alignment_score']
    
    # Intraday Momentum Persistence Patterns
    # Morning Session Momentum
    data['morning_momentum'] = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['morning_persistence'] = data['morning_momentum'].rolling(window=3).apply(
        lambda x: np.mean(np.diff(np.sign(x.dropna()))) if len(x.dropna()) > 1 else 0, raw=False
    )
    
    # Afternoon Session Momentum
    data['afternoon_momentum'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['afternoon_persistence'] = data['afternoon_momentum'].rolling(window=5).apply(
        lambda x: np.mean(np.diff(np.sign(x.dropna()))) if len(x.dropna()) > 1 else 0, raw=False
    )
    
    # Full Day Momentum Profile
    data['full_day_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Momentum concentration
    data['momentum_concentration'] = np.maximum(
        np.abs(data['morning_momentum']), 
        np.abs(data['afternoon_momentum'])
    )
    
    # Session transfer correlation
    data['session_transfer'] = data['morning_momentum'].rolling(window=5).corr(data['afternoon_momentum'])
    
    # Combined intraday momentum
    data['combined_momentum'] = (
        data['morning_momentum'] * (1 - np.abs(data['morning_persistence'])) +
        data['afternoon_momentum'] * (1 - np.abs(data['afternoon_persistence'])) +
        data['full_day_momentum']
    ) * data['momentum_concentration']
    
    # Trade Size Distribution Dynamics
    data['trade_size'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['trade_size_ma_5d'] = data['trade_size'].rolling(window=5).mean()
    data['trade_size_change'] = data['trade_size'].pct_change()
    data['trade_size_volatility'] = data['trade_size'].rolling(window=10).std()
    
    # Trade size regimes
    data['trade_size_regime'] = pd.cut(
        data['trade_size'] / data['trade_size_ma_5d'], 
        bins=[-np.inf, 0.8, 1.2, np.inf], 
        labels=[-1, 0, 1]
    ).astype(float)
    
    # Trade size-price correlation
    data['trade_size_price_corr'] = data['trade_size'].rolling(window=10).corr(data['close'].pct_change())
    
    # Trade size dynamics filtering
    data['trade_size_filter'] = np.where(
        data['trade_size_regime'] == 1, 1.2,
        np.where(data['trade_size_regime'] == -1, 0.8, 1.0)
    ) * (1 - data['trade_size_volatility'] / data['trade_size_volatility'].rolling(window=20).mean())
    
    # Synthesize Composite Alpha Factor
    # Combine divergence with intraday momentum
    data['divergence_momentum'] = data['weighted_divergence'] * data['combined_momentum']
    
    # Apply trade size filtering
    data['filtered_signal'] = data['divergence_momentum'] * data['trade_size_filter']
    
    # Multi-timeframe consistency framework
    data['final_factor'] = (
        data['filtered_signal'] * 
        data['alignment_score'] * 
        (1 - np.abs(data['session_transfer'])) *  # Lower correlation suggests stronger timing
        np.sign(data['trade_size_price_corr'])  # Use correlation sign as confirmation
    )
    
    return data['final_factor']
