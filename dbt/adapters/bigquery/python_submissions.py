import uuid
import nbformat
import re
import os

from typing import Dict, Union

from dbt.adapters.events.logging import AdapterLogger

from dbt.adapters.base import PythonJobHelper
from google.api_core.future.polling import POLLING_PREDICATE

from dbt.adapters.bigquery import BigQueryConnectionManager, BigQueryCredentials
from google.api_core import retry
from google.api_core.client_options import ClientOptions
from google.cloud import storage, dataproc_v1  # type: ignore
from google.cloud.dataproc_v1.types.batches import Batch
from google.cloud import aiplatform_v1beta1

from dbt.adapters.bigquery.dataproc.batch import (
    create_batch_request,
    poll_batch_job,
    DEFAULT_JAR_FILE_URI,
    update_batch_from_config,
)

OPERATION_RETRY_TIME = 10
logger = AdapterLogger("BigQuery")

DEFAULT_VAI_NOTEBOOK_NAME = "Default"
os.environ["GRPC_VERBOSITY"] = "NONE"


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
            "dataproc_region",
            "gcs_bucket",
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
        self.gcs_location = "gs://{}/{}".format(self.credential.gcs_bucket, self.model_file_name)

        # set retry policy, default to timeout after 24 hours
        self.timeout = self.parsed_model["config"].get(
            "timeout", self.credential.job_execution_timeout_seconds or 60 * 60 * 24
        )
        self.result_polling_policy = retry.Retry(
            predicate=POLLING_PREDICATE, maximum=10.0, timeout=self.timeout
        )
        self.client_options = ClientOptions(
            api_endpoint="{}-dataproc.googleapis.com:443".format(self.credential.dataproc_region)
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


class ClusterDataprocHelper(BaseDataProcHelper):
    def _get_job_client(self) -> dataproc_v1.JobControllerClient:
        if not self._get_cluster_name():
            raise ValueError(
                "Need to supply dataproc_cluster_name in profile or config to submit python job with cluster submission method"
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


class BigFramesHelper(PythonJobHelper):
    def __init__(self, parsed_model: Dict, credential: BigQueryCredentials) -> None:
        """_summary_
        Args:
            credential (_type_): _description_
        """

        schema = parsed_model["schema"]
        identifier = parsed_model["alias"]
        self.parsed_model = parsed_model
        python_required_configs = [
            "bigframes_region",
            "gcs_bucket",
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
        self.ai_platform_client = aiplatform_v1beta1.NotebookServiceClient(
            credentials=self.GoogleCredentials,
            client_options=ClientOptions(
                api_endpoint=f"{self.credential.bigframes_region}-aiplatform.googleapis.com"
            ),
        )
        if not getattr(credential, "bigframes_upload_notebook_gcs"):
            self.credential.bigframes_upload_notebook_gcs = False
        self.gcs_location = "gs://{}/{}".format(self.credential.gcs_bucket, self.model_file_name)
        self.timeout = self.parsed_model["config"].get(
            "timeout", self.credential.job_execution_timeout_seconds or 60 * 60 * 24
        )
        self.result_polling_policy = retry.Retry(
            predicate=POLLING_PREDICATE, maximum=10.0, timeout=self.timeout
        )

    def _get_batch_id(self) -> str:
        model = self.parsed_model
        default_batch_id = str(uuid.uuid4())
        return model["config"].get("batch_id", default_batch_id)

    def _upload_to_gcs(self, filename: str, compiled_code: str) -> None:
        bucket = self.storage_client.get_bucket(self.credential.gcs_bucket)
        blob = bucket.blob(filename)
        blob.upload_from_string(compiled_code)

    def _py_to_ipynb(self, compiled_code):
        nb = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_code_cell(compiled_code))
        return nbformat.writes(nb, nbformat.NO_CONVERT)

    def submit(self, compiled_code: str):
        notebook_compiled_code = self._py_to_ipynb(compiled_code)
        notebook_template_id = self._get_notebook_template_id(
            self.credential.bigframes_notebook_template_id
        )
        if self.credential.bigframes_upload_notebook_gcs:
            self._upload_to_gcs(self.model_file_name, notebook_compiled_code)
        return self._submit_bigframes_job(notebook_compiled_code, notebook_template_id)

    def _get_notebook_template_id(self, notebook_template_id):
        if not notebook_template_id:
            # If not template id is specified, use the Default one
            request = aiplatform_v1beta1.ListNotebookRuntimeTemplatesRequest(
                parent=f"projects/{self.credential.execution_project}/locations/{self.credential.bigframes_region}",
                filter=f'display_name = "{DEFAULT_VAI_NOTEBOOK_NAME}"',
            )
            page_result = self.ai_platform_client.list_notebook_runtime_templates(request=request)
            if page_result:
                # Extract template id from name
                notebook_template_id = re.search(
                    r"notebookRuntimeTemplates/(\d+)", next(iter(page_result)).name
                ).group(1)
                return notebook_template_id
            else:
                raise ValueError("No notebook runtime templates found, even Default.")
        else:
            return notebook_template_id

    def _submit_bigframes_job(self, notebook_compiled_code, notebook_template_id):
        notebook_execution_job = aiplatform_v1beta1.NotebookExecutionJob()
        notebook_execution_job.notebook_runtime_template_resource_name = f"projects/{self.credential.execution_project}/locations/{self.credential.bigframes_region}/notebookRuntimeTemplates/{notebook_template_id}"

        if self.credential.bigframes_upload_notebook_gcs:
            notebook_execution_job.gcs_notebook_source = (
                aiplatform_v1beta1.NotebookExecutionJob.GcsNotebookSource(uri=self.gcs_location)
            )
        else:
            # Exec directly from raw bytes
            notebook_compiled_code_bytes = notebook_compiled_code.encode("utf-8")
            notebook_execution_job.direct_notebook_source = (
                aiplatform_v1beta1.NotebookExecutionJob.DirectNotebookSource(
                    content=notebook_compiled_code_bytes
                )
            )
        notebook_execution_job.service_account = self.GoogleCredentials._service_account_email
        notebook_execution_job.gcs_output_uri = "gs://{}/{}/logs".format(
            self.credential.gcs_bucket, self.model_file_name
        )
        notebook_execution_job.display_name = self._get_batch_id()
        request = aiplatform_v1beta1.CreateNotebookExecutionJobRequest(
            parent=f"projects/{self.credential.execution_project}/locations/{self.credential.bigframes_region}",
            notebook_execution_job=notebook_execution_job,
        )
        _ = self.ai_platform_client.create_notebook_execution_job(request=request).result()
