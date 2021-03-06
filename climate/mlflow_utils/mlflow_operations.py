import os

import mlflow
from climate.blob_storage_operations.blob_operations import Blob_Operation
from mlflow.tracking import MlflowClient
from utils.logger import App_Logger
from utils.model_utils import Model_Utils
from utils.read_params import read_params


class MLFlow_Operations:
    """
    Description :    This class shall be used for handling all the mlflow operations

    Version     :    1.2
    Revisions   :    moved to setup to cloud
    """

    def __init__(self, db_name, collection_name):
        self.config = read_params()

        self.class_name = self.__class__.__name__

        self.log_writer = App_Logger()

        self.model_utils = Model_Utils()

        self.blob = Blob_Operation()

        self.db_name = db_name

        self.collection_name = collection_name

        self.mlflow_save_format = self.config["mlflow_config"]["serialization_format"]

        self.trained_models_dir = self.config["models_dir"]["trained"]

        self.staged_models_dir = self.config["models_dir"]["stag"]

        self.prod_models_dir = self.config["models_dir"]["prod"]

        self.model_save_format = self.config["model_utils"]["save_format"]

    def get_experiment_from_mlflow(self, exp_name):
        """
        Method Name :   get_experiment_from_mlflow
        Description :   This method gets the experiment from mlflow by name

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_experiment_from_mlflow.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            exp = mlflow.get_experiment_by_name(name=exp_name)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_info=f"Got {exp_name} experiment from mlflow",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return exp

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_runs_from_mlflow(self, exp_id):
        """
        Method Name :   get_runs_from_mlflow
        Description :   This method gets the runs from mlflow as dataframe

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_runs_from_mlflow.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            runs = mlflow.search_runs(experiment_ids=exp_id)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_info=f"Completed searchiing for runs in mlflow with experiment ids as {exp_id}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return runs

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def set_mlflow_experiment(self, experiment_name):
        """
        Method Name :   set_mlflow_experiment
        Description :   This method sets the mlflow experiment witht he given name

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.set_mlflow_experiment.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            mlflow.set_experiment(experiment_name=experiment_name)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_info=f"Set mlflow experiment with name as {experiment_name}",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_mlflow_client(self, server_uri):
        """
        Method Name :   get_mlflow_client
        Description :   This method gets the mlflow client with the particular server uri

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_mlflow_client.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            client = MlflowClient(tracking_uri=server_uri)

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_info="Got mlflow client with tracking uri",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return client

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_remote_server_uri(self):
        """
        Method Name :   get_remote_server_uri
        Description :   This method sets the mlflow client with the particular server uri

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_remote_server_uri.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            remote_server_uri = os.environ["MLFLOW_TRACKING_URI"]

            self.log_writer.log(
                collection_name=self.collection_name,
                log_info="Got mlflow tracking uri",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return remote_server_uri

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def set_mlflow_tracking_uri(self):
        """
        Method Name :   set_mlflow_tracking_uri
        Description :   This method sets the mlflow client with the particular server uri

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.set_mlflow_tracking_uri.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            server_uri = self.get_remote_server_uri()

            mlflow.set_tracking_uri(server_uri)

            self.log_writer.log(
                collection_name=self.collection_name,
                log_info="Set mlflow tracking uri",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def get_mlflow_models(self):
        """
        Method Name :   get_mlflow_models
        Description :   This method gets the mlflow models from the mlflow model registry

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.get_mlflow_models.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            remote_server_uri = self.get_remote_server_uri()

            client = self.get_mlflow_client(server_uri=remote_server_uri)

            reg_model_names = [rm.name for rm in client.list_registered_models()]

            self.log_writer.log(
                collection_name=self.collection_name,
                log_info="Got registered model from mlflow",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return reg_model_names

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def search_mlflow_models(self, order):
        """
        Method Name :   search_mlflow_models
        Description :   This method searches the mlflow models and returns the result in the given order

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.search_mlflow_models.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            remote_server_uri = self.get_remote_server_uri()

            client = self.get_mlflow_client(server_uri=remote_server_uri)

            results = client.search_registered_models(order_by=[f"name {order}"])

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_info=f"Got registered models in mlflow in {order} order",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            return results

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def log_model(self, model, model_name):
        """
        Method Name :   log_model
        Description :   This method logs the model to mlflow with the mentioned format and name

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.log_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                serialization_format=self.mlflow_save_format,
                registered_model_name=model_name,
                artifact_path=model_name,
            )

            self.log_writer.log(
                collection_name=self.collection_name,
                log_info=f"Logged {model_name} model in mlflow",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def log_metric(self, model_name, metric):
        """
        Method Name :   log_metric
        Description :   This method logs the metric of model to mlflow

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.log_metric.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            model_score_name = f"{model_name}-best_score"

            mlflow.log_metric(key=model_score_name, value=metric)

            self.log_writer.log(
                collection_name=self.collection_name,
                log_info=f"{model_score_name} logged in mlflow",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def log_param(self, idx, model, model_name, param):
        """
        Method Name :   log_param
        Description :   This method logs the params of the model to mlflow

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.log_param.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            model_param_name = model_name + str(idx) + f"-{param}"

            mlflow.log_param(key=model_param_name, value=model.__dict__[param])

            self.log_writer.log(
                collection_name=self.collection_name,
                log_info=f"{model_param_name} logged in mlflow",
            )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def log_all_for_model(self, idx, model, model_param_name, model_score):
        """
        Method Name :   log_all_for_model
        Description :   This method logs the params,metrics and model itself for the particular model

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.log_all_for_model.__name__

        try:
            self.log_writer.start_log(
                key="start",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

            base_model_name = self.model_utils.get_model_name(
                model=model, db_name=self.db_name, collection_name=self.collection_name
            )

            if base_model_name is "KMeans":
                self.log_model(model=model, model_name=base_model_name)

            else:
                model_name = base_model_name + str(idx)

                self.log_writer.log(
                    collection_name=self.collection_name,
                    log_info=f"Got the model name as {model_name}",
                )

                model_params_list = list(
                    self.config["model_params"][model_param_name].keys()
                )

                self.log_writer.log(
                    collection_name=self.collection_name,
                    log_info=f"Created a list of params based on {model_param_name}",
                )

                for param in model_params_list:
                    self.log_param(
                        idx=idx,
                        model=model,
                        model_name=model_name,
                        param=param,
                    )

                self.log_model(model=model, model_name=model_name)

                self.log_metric(model_name=model_name, metric=float(model_score))

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

    def transition_mlflow_model(
        self, model_version, stage, model_name, from_container_name, to_container_name
    ):
        """
        Method Name :   transition_mlflow_model
        Description :   This method transitions the models in mlflow and as well as in blob container based on
                        the best model for the particular cluster

        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        method_name = self.transition_mlflow_model.__name__

        self.log_writer.start_log(
            key="start",
            class_name=self.class_name,
            method_name=method_name,
            db_name=self.db_name,
            collection_name=self.collection_name,
        )

        try:
            remote_server_uri = self.get_remote_server_uri()

            current_version = model_version

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_info=f"Got {current_version} as the current model version",
            )

            client = self.get_mlflow_client(server_uri=remote_server_uri)

            trained_model_file = (
                self.trained_models_dir + "/" + model_name + self.model_save_format
            )

            stag_model_file = (
                self.staged_models_dir + "/" + model_name + self.model_save_format
            )

            prod_model_file = (
                self.prod_models_dir + "/" + model_name + self.model_save_format
            )

            self.log_writer.log(
                db_name=self.db_name,
                collection_name=self.collection_name,
                log_info="Created trained,stag and prod model files",
            )

            if stage == "Production":
                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_info=f"{stage} is selected for transition",
                )

                client.transition_model_version_stage(
                    name=model_name, version=current_version, stage=stage
                )

                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_info=f"Transitioned {model_name} to {stage} in mlflow",
                )

                self.blob.copy_data(
                    from_file_name=trained_model_file,
                    from_container_name=from_container_name,
                    to_file_name=prod_model_file,
                    to_container_name=to_container_name,
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                )

            elif stage == "Staging":
                self.log_writer.log(
                    collection_name=self.collection_name,
                    log_info=f"{stage} is selected for transition",
                )

                client.transition_model_version_stage(
                    name=model_name, version=current_version, stage=stage
                )

                self.log_writer.log(
                    collection_name=self.collection_name,
                    log_info=f"Transitioned {model_name} to {stage} in mlflow",
                )

                self.blob.copy_data(
                    from_file_name=trained_model_file,
                    from_container_name=from_container_name,
                    to_file_name=stag_model_file,
                    to_container_name=to_container_name,
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                )

            else:
                self.log_writer.log(
                    db_name=self.db_name,
                    collection_name=self.collection_name,
                    log_info="Please select stage for model transition",
                )

            self.log_writer.start_log(
                key="exit",
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )

        except Exception as e:
            self.log_writer.exception_log(
                error=e,
                class_name=self.class_name,
                method_name=method_name,
                db_name=self.db_name,
                collection_name=self.collection_name,
            )
