import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def build_pipeline():
    """constructs the robust pre-processor and model architecture"""
    num_features = ['months_listed', 'bedrooms', 'area']
    cat_features = ['city', 'house_type']
    
    # numeric transformations
    # fallback to mean if NaNs appear, then standardise
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # categorical transformations
    # fallback to most frequent, then one-hot encode
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # combine into a single pre-processor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
        
    # append the random forest estimator
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=21))
    ])
    
    return model_pipeline, num_features, cat_features
    
def main():
    print('Loading datasets...')
    # paths are set relative to repository root
    train_data = pd.read_csv('data/train.csv')
    val_data = pd.read_csv('data/validation.csv')
    
    target = 'sale_price'
    model_pipeline, num_features, cat_features = build_pipeline()
    
    expected_features = num_features+cat_features
    X = train_data[expected_features]
    y = train_data[target]
    
    # local evaluation
    # (proves the model works and gives a metric)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    print('Evaluating pipeline on local split...')
    model_pipeline.fit(X_train, y_train)
    local_preds = model_pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, local_preds)**0.5
    print(f'Local validation RMSE: ${rmse:,.2f}')

    # retrain on all available train data for accuracy
    print('Re-fitting on full training dataset...')
    model_pipeline.fit(X, y)
    
    # predict on unseen hold-out set (based on cert assessment specs)
    print('Generating predictions for validation set...')
    X_val = val_data[expected_features]
    val_data['predicted_price'] = model_pipeline.predict(X_val)
    
    # export answers and model
    print('Exporting to disk...')
    val_data.to_csv('data/predictions.csv', index=False)
    print('Exported predictions to: data/predictions.csv')
    joblib.dump(model_pipeline, 'models/rnd_forest_pipeline.pkl')
    print('Export model to: models/rnd_forest_pipeline.pkl')
    
if __name__ == '__main__':
    main()
    