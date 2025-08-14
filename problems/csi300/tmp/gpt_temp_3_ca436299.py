import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['High-Low_Spread'] = df['high'] - df['low']
    df['Close-Open_Spread'] = df['close'] - df['open']
    df['Intraday_Momentum'] = df['High-Low_Spread'] - df['Close-Open_Spread']
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = df['amount'] / df['volume']
    
    # Determine Volume Synchronization
    df['Log_Volume_Change'] = np.log(df['volume'] / df['volume'].shift(1))
    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Integrate Price and Volume Dynamics
    df['Integrated_Indicator'] = df['Intraday_Momentum'] * df['Log_Return']
    df['Combined_Indicator'] = df['Integrated_Indicator'] + df['VWAP']
    
    # Enhance Factor with Intraday and Relative Strength
    df['Intraday_Trend'] = (df['high'] - df['low']) / df['low']
    df['Rolling_Avg_Close'] = df['close'].rolling(window=30).mean()
    df['Relative_Strength'] = df['close'] / df['Rolling_Avg_Close']
    
    # Incorporate Market Microstructure
    # Assuming we have buy_volume and sell_volume columns
    df['Tick_Imbalance'] = df['buy_volume'] - df['sell_volume']
    # Assuming we have bid_volume and ask_volume columns
    df['Order_Book_Imbalance'] = df['bid_volume'] - df['ask_volume']
    
    # Prepare data for PCA
    indicators = df[['Integrated_Indicator', 'VWAP', 'Intraday_Trend', 'Relative_Strength', 'Tick_Imbalance', 'Order_Book_Imbalance']].dropna()
    
    # Apply Principal Component Analysis (PCA) on Indicators
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(indicators)
    
    # Assign the first principal component as the final alpha factor
    final_alpha_factor = pd.Series(principal_components.flatten(), index=indicators.index, name='Final_Alpha_Factor')
    
    return final_alpha_factor
