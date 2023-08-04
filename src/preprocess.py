import numpy as np
import pandas as pd
# from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

np
pd
# TargetEncoder

CAT_FEATURES = [
]

def func1():
    return

preprocess_pipeline = ColumnTransformer(
    transformers = [
        ("transformation1", FunctionTransformer(func1), ["col1"]),
        # ("target_encoding", TargetEncoder(), CAT_FEATURES),
    ]
)