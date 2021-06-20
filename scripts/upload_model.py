import os
from src.main.util.model_downloader_seafile import get_write_account_data, SeafileModelSyncer, model_downloader_logger
from src.static import settings
from main.util.logutils import setup_logging

setup_logging("INFO")
localpath = settings.DATA_BASE
account, password, server, repoid, repopath, modelversions = get_write_account_data()
modelsyncer = SeafileModelSyncer(server, account, password, repoid, repopath)
if modelsyncer.repo is not None:
    print("Do you really want to upload the following:")
    for mname, mversion in modelversions.items():
        if mversion is not None:
            print(f"{mname} in version {mversion}")
    if input("? [y/n]").lower() == "y":
        modelsyncer.upload_modeldirs(localpath, modelversions, overwrite_version=False)