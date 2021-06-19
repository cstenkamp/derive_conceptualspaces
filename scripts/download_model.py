import os
from src.main.util.model_downloader_seafile import get_read_account_data, SeafileModelSyncer, model_downloader_logger
from src.static import settings

model_downloader_logger.setLevel("INFO")
localpath = settings.DATA_BASE
account, password, server, repoid, repopath, modelversions = get_read_account_data()
modelsyncer = SeafileModelSyncer(server, account, password, repoid, repopath)
if modelsyncer.repo is not None:
    modelsyncer.download_modeldirs(localpath, modelversions, force_delete=False, fail_if_absent=True)