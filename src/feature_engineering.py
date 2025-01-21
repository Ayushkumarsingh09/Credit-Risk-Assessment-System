from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def transform_features(data, numerical_features, categorical_features):
    """
    Transforms features using standard scaling and one-hot encoding.
    """
    # Pipeline for numerical features
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    
    X_transformed = preprocessor.fit_transform(data)
    return X_transformed, preprocessor
