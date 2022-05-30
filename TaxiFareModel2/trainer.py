# imports
from termcolor import colored
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from memoized_property import memoized_property

import numpy as np
import mlflow
import joblib



from TaxiFareModel2.encoders import *
from TaxiFareModel2.data import *

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[UK] [London] [ywang] taxifare v2"

class Trainer(object):
    EXPERIMENT_NAME = "TaxifareModel"
    ESTIMATOR = "Linear"
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.kwargs = kwargs
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)
        self.X = X
        self.y = y

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "Lasso":
            model = Lasso()
        elif estimator == "Ridge":
            model = Ridge()
        elif estimator == "Linear":
            model = LinearRegression()
        elif estimator == "GBM":
            model = GradientBoostingRegressor()
        elif estimator == "RandomForest":
            model = RandomForestRegressor()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'max_features': ['auto', 'sqrt']}
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        elif estimator == "xgboost":
            model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=10, learning_rate=0.05,
                                 gamma=3)
            self.model_params = {'max_depth': range(10, 20, 2),
                                 'n_estimators': range(60, 220, 40),
                                 'learning_rate': [0.1, 0.01, 0.05]
                                 }
        else:
            model = Lasso()
        estimator_params = self.kwargs.get("estimator_params", {})
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])

        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        pipe = Pipeline([
        ('preproc', preproc_pipe),
        ('rgs', self.get_estimator())
        ])

        return pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test)**2).mean())
        print(rmse)
        return rmse

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":

    experiment = 'test1'
    params = dict(nrows=1000,
                  estimator="xgboost",
                  experiment_name=experiment)
    # get data
    print("############   Loading Data   ############")
    df = get_data(**params)

    # clean data
    df = clean_data(df)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # train
    trainer = Trainer(X_train, y_train, **params
                      )
    print(colored("############  Training model   ############", "red"))
    pipeline = trainer.run()

    # evaluate
    print(colored("############  Evaluating model ############", "blue"))
    rmse = trainer.evaluate(X_val, y_val)
    print(f"rmse: {rmse}")
    print(colored("############   Saving model    ############", "green"))
    trainer.save_model()
