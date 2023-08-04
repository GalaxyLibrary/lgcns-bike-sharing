import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


# TargetEncoder

CAT_FEATURES = ["weather","season"]

def func1(df):
    weather={1:'clear',2:'mist',3:'light',4:'heavy'}
    season={ 1 : 'spring', 2 : 'summer', 3 : 'fall', 4 : 'winter'} 
    df['weather'] = df.apply(lambda x: weather[int(x)])
    df['season'] = df.apply(lambda x: season[int(x)])
    return df

preprocess_pipeline = ColumnTransformer(
    transformers = [
        ("transformation1", FunctionTransformer(func1), CAT_FEATURES),
        ("one_hot", OneHotEncoder(), CAT_FEATURES),
        ("count_log", FunctionTransformer(np.log), ['count']),

        # ("target_encoding", TargetEncoder(), CAT_FEATURES),
    ]
)