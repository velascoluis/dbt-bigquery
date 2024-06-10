import uuid
from typing import Dict, Union

from dbt.adapters.events.logging import AdapterLogger

from dbt.adapters.base import PythonJobHelper
from google.api_core.future.polling import POLLING_PREDICATE

from dbt.adapters.bigquery import BigQueryConnectionManager, BigQueryCredentials
from google.api_core import retry
from google.api_core.client_options import ClientOptions
from google.cloud import storage, dataproc_v1  # type: ignore
from google.cloud.dataproc_v1.types.batches import Batch
from google.cloud import bigquery_connection_v1

from dbt.adapters.bigquery.dataproc.batch import (
    create_batch_request,
    poll_batch_job,
    DEFAULT_JAR_FILE_URI,
    update_batch_from_config,
)

OPERATION_RETRY_TIME = 10
logger = AdapterLogger("BigQuery")


class BaseDataProcHelper(PythonJobHelper):
    def __init__(self, parsed_model: Dict, credential: BigQueryCredentials) -> None:
        """_summary_

        Args:
            credential (_type_): _description_
        """
        # validate all additional stuff for python is set
        schema = parsed_model["schema"]
        identifier = parsed_model["alias"]
        self.parsed_model = parsed_model
        python_required_configs = [
            "gcs_bucket"
        ]
        for required_config in python_required_configs:
            if not getattr(credential, required_config):
                raise ValueError(
                    f"Need to supply {required_config} in profile to submit python job"
                )
        self.model_file_name = f"{schema}/{identifier}.py"
        self.credential = credential
        self.GoogleCredentials = BigQueryConnectionManager.get_credentials(credential)
        self.storage_client = storage.Client(
            project=self.credential.execution_project,
            credentials=self.GoogleCredentials,
        )
        self.gcs_location = "gs://{}/{}".format(
            self.credential.gcs_bucket, self.model_file_name
        )

        # set retry policy, default to timeout after 24 hours
        self.timeout = self.parsed_model["config"].get(
            "timeout", self.credential.job_execution_timeout_seconds or 60 * 60 * 24
        )
        self.result_polling_policy = retry.Retry(
            predicate=POLLING_PREDICATE, maximum=10.0, timeout=self.timeout
        )
        self.job_client = self._get_job_client()

    def _upload_to_gcs(self, filename: str, compiled_code: str) -> None:
        bucket = self.storage_client.get_bucket(self.credential.gcs_bucket)
        blob = bucket.blob(filename)
        blob.upload_from_string(compiled_code)

    def submit(self, compiled_code: str) -> dataproc_v1.types.jobs.Job:
        # upload python file to GCS
        self._upload_to_gcs(self.model_file_name, compiled_code)
        # submit dataproc job
        return self._submit_dataproc_job()

    def _get_job_client(
        self,
    ) -> Union[dataproc_v1.JobControllerClient, dataproc_v1.BatchControllerClient]:
        raise NotImplementedError("_get_job_client not implemented")

    def _submit_dataproc_job(self) -> dataproc_v1.types.jobs.Job:
        raise NotImplementedError("_submit_dataproc_job not implemented")


class DataProcStoredProcedureHelper(BaseDataProcHelper):
    def _submit_dataproc_job(self):
        python_required_configs = ["spark_external_connection_name"]
        for required_config in python_required_configs:
            if not getattr(self.credential, required_config):
                raise ValueError(
                    f"Need to supply {required_config} in profile to submit python job"
                )

        database = self.parsed_model["database"]
        schema = self.parsed_model["schema"]
        conn_name = (
            self.credential.spark_external_connection_name
            or f"spark-dbt-{database}-{schema}"
        )
        conn_name = self._get_or_create_ext_conn(conn_name)
        sproc_name, sproc_ddl = self._get_stored_procedure_ddl(conn_name)
        sproc_call = self._get_call_stored_procedure(sproc_name)
        client = BigQueryConnectionManager.get_bigquery_client(self.credential)
        client.query_and_wait(sproc_ddl)
        client.query_and_wait(sproc_call)

    def _get_uuid(self):
        return str(uuid.uuid4()).replace("-", "_")

    def _get_or_create_ext_conn(self, conn_name):
        conn_client = bigquery_connection_v1.ConnectionServiceClient()
        database = self.parsed_model["database"]
        region = self.credential.location
        parent = f"projects/{database}/locations/{region}"
        request = bigquery_connection_v1.ListConnectionsRequest(parent=parent)
        existing_connections = conn_client.list_connections(request=request)

        for conn in existing_connections:
            if conn.name.rsplit("/", 1)[-1] == conn_name:
                return conn_name

        connection = bigquery_connection_v1.Connection(
            friendly_name=conn_name, spark=bigquery_connection_v1.SparkProperties()
        )
        # This will create a new connection, but still needs to grant reader permissions on the bucket where
        # the code is stored
        response = conn_client.create_connection(
            request={
                "parent": parent,
                "connection_id": conn_name,
                "connection": connection,
            }
        )
        return response.name.rsplit("/", 1)[-1]

    def _get_stored_procedure_ddl(self, conn_name):
        schema = self.parsed_model["schema"]
        database = self.parsed_model["database"]
        identifier = self.parsed_model["alias"]
        gcs_code_location_uri = self.gcs_location
        uuid = self._get_uuid()
        sproc_name = f"{database}.{schema}.{identifier}__{uuid}"
        sproc_ddl = f"""
        CREATE PROCEDURE `{sproc_name}`()
        EXTERNAL SECURITY INVOKER
        WITH CONNECTION `{database}.{self.credential.location}.{conn_name}`
        OPTIONS(engine="SPARK", runtime_version="1.1", main_file_uri="{gcs_code_location_uri}")
        LANGUAGE PYTHON"""
        return sproc_name, sproc_ddl

    def _get_call_stored_procedure(self, sproc_name):
        # This would enforce using the dbt SA account
        schema = self.parsed_model["schema"]
        return f"""
        SET @@spark_proc_properties.staging_bucket='{self.credential.gcs_bucket}';
        SET @@spark_proc_properties.staging_dataset_id='{schema}';
        SET @@spark_proc_properties.service_account='{self.GoogleCredentials._service_account_email}';
        CALL `{sproc_name}`()
        """

    def _get_job_client(self):
        pass


class ClusterDataprocHelper(BaseDataProcHelper):
    def _get_job_client(self) -> dataproc_v1.JobControllerClient:
        if not self._get_cluster_name():
            raise ValueError(
                "Need to supply dataproc_cluster_name in profile or config to submit python job with cluster submission method"
            )
        python_required_configs = [
            "dataproc_region"
        ]
        for required_config in python_required_configs:
            if not getattr(self.credential, required_config):
                raise ValueError(
                    f"Need to supply {required_config} in profile to submit python job"
                )
        self.client_options = ClientOptions(
            api_endpoint="{}-dataproc.googleapis.com:443".format(
                self.credential.dataproc_region
            )
        )
        return dataproc_v1.JobControllerClient(  # type: ignore
            client_options=self.client_options, credentials=self.GoogleCredentials
        )

    def _get_cluster_name(self) -> str:
        return self.parsed_model["config"].get(
            "dataproc_cluster_name", self.credential.dataproc_cluster_name
        )

    def _submit_dataproc_job(self) -> dataproc_v1.types.jobs.Job:
        job = {
            "placement": {"cluster_name": self._get_cluster_name()},
            "pyspark_job": {
                "main_python_file_uri": self.gcs_location,
            },
        }
        operation = self.job_client.submit_job_as_operation(  # type: ignore
            request={
                "project_id": self.credential.execution_project,
                "region": self.credential.dataproc_region,
                "job": job,
            }
        )
        # check if job failed
        response = operation.result(polling=self.result_polling_policy)
        if response.status.state == 6:
            raise ValueError(response.status.details)
        return response


class ServerlessDataProcHelper(BaseDataProcHelper):
    def _get_job_client(self) -> dataproc_v1.BatchControllerClient:
        python_required_configs = [
            "dataproc_region"
        ]
        for required_config in python_required_configs:
            if not getattr(self.credential, required_config):
                raise ValueError(
                    f"Need to supply {required_config} in profile to submit python job"
                )
        self.client_options = ClientOptions(
            api_endpoint="{}-dataproc.googleapis.com:443".format(
                self.credential.dataproc_region
            )
        )
        return dataproc_v1.BatchControllerClient(
            client_options=self.client_options, credentials=self.GoogleCredentials
        )

    def _get_batch_id(self) -> str:
        model = self.parsed_model
        default_batch_id = str(uuid.uuid4())
        return model["config"].get("batch_id", default_batch_id)

    def _submit_dataproc_job(self) -> Batch:
        batch_id = self._get_batch_id()
        logger.info(f"Submitting batch job with id: {batch_id}")
        request = create_batch_request(
            batch=self._configure_batch(),
            batch_id=batch_id,
            region=self.credential.dataproc_region,  # type: ignore
            project=self.credential.execution_project,  # type: ignore
        )  # type: ignore
        # make the request
        self.job_client.create_batch(request=request)  # type: ignore
        return poll_batch_job(
            parent=request.parent,
            batch_id=batch_id,
            job_client=self.job_client,  # type: ignore
            timeout=self.timeout,
        )
        # there might be useful results here that we can parse and return
        # Dataproc job output is saved to the Cloud Storage bucket
        # allocated to the job. Use regex to obtain the bucket and blob info.
        # matches = re.match("gs://(.*?)/(.*)", response.driver_output_resource_uri)
        # output = (
        #     self.storage_client
        #     .get_bucket(matches.group(1))
        #     .blob(f"{matches.group(2)}.000000000")
        #     .download_as_string()
        # )

    def _configure_batch(self):
        # create the Dataproc Serverless job config
        # need to pin dataproc version to 1.1 as it now defaults to 2.0
        # https://cloud.google.com/dataproc-serverless/docs/concepts/properties
        # https://cloud.google.com/dataproc-serverless/docs/reference/rest/v1/projects.locations.batches#runtimeconfig
        batch = dataproc_v1.Batch(
            {
                "runtime_config": dataproc_v1.RuntimeConfig(
                    version="1.1",
                    properties={
                        "spark.executor.instances": "2",
                    },
                )
            }
        )
        # Apply defaults
        batch.pyspark_batch.main_python_file_uri = self.gcs_location
        jar_file_uri = self.parsed_model["config"].get(
            "jar_file_uri",
            DEFAULT_JAR_FILE_URI,
        )
        batch.pyspark_batch.jar_file_uris = [jar_file_uri]

        # Apply configuration from dataproc_batch key, possibly overriding defaults.
        if self.credential.dataproc_batch:
            batch = update_batch_from_config(self.credential.dataproc_batch, batch)
        return batch
