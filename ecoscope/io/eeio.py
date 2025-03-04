import logging

import ee

logger = logging.getLogger(__name__)


class EarthEngineIO:
    def __init__(self, **kwargs):
        self._service_account = kwargs.get("service_account")
        self._private_key = kwargs.get("private_key")
        self._private_key_file = kwargs.get("private_key_file")
        self._ee_project = kwargs.get("ee_project")

        if self._service_account:
            credentials = ee.ServiceAccountCredentials(
                email=self._service_account,
                key_data=self._private_key,
                key_file=self._private_key_file,
            )
            ee.Initialize(credentials)
        else:
            logger.info("No service account is set up. Please authenticate manually.")
            ee.Authenticate()
            ee.Initialize(project=self._ee_project)

        logger.info("Successfully connected to EarthEngine.")
