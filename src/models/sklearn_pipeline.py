from sklearn.pipeline import Pipeline
import wandb
import mlflow


class PipelineSklearn:
    def __init__(self, preprocessor, feature_extractor, model, k_folds):
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.model = model
        self.k_folds = k_folds
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('feature extractor', self.feature_extractor),
            ('model', self.model)
        ])

    def train(self, X, y, wandb_api, wandb_project):
        """
        Cross-validation training with mlflow and wandb autologging.
        :param X:
        :param y:
        :param wandb_api: API for wandb project
        :param wandb_project: wandb project name
        :return:
        """

        mlflow.sklearn.autolog()
        wandb.login(key=wandb_api)
        wandb.init(project=wandb_project)

        with mlflow.start_run():
            for train_idx, test_idx in self.k_folds.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                self.pipeline.fit(X_train, y_train)
                wandb.sklearn.plot_summary_metrics(self.pipeline, X_train, X_test, y_train, y_test)


