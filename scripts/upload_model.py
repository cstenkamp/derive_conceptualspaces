import argparse

from src.main.util.model_downloader_seafile import get_write_account_data, SeafileModelSyncer, model_downloader_logger
from src.static import settings
from src.main.util.logutils import setup_logging




def parse_command_line_args(default_versions):
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--versions',
                        help='If you want to overwrite the settings.py in terms of which versions you want to upload',
                        default=default_versions)
    parser.add_argument('-o', '--overwrite', default=False, help='If you want to overwrite the version',
                        action='store_true')
    return parser.parse_args()


def main():
    setup_logging("INFO")
    localpath = settings.DATA_BASE
    account, password, server, repoid, repopath, modelversions = get_write_account_data()
    args = parse_command_line_args(modelversions)
    modelversions = eval(args.versions) if isinstance(args.versions, str) else args.versions
    modelsyncer = SeafileModelSyncer(server, account, password, repoid, repopath)
    if modelsyncer.repo is not None:
        print("Do you really want to upload the following:")
        for mname, mversion in modelversions.items():
            if mversion is not None:
                print(f"{mname} in version {mversion}")
        if input("? [y/n]").lower() == "y":
            modelsyncer.upload_modeldirs(localpath, modelversions, overwrite_version=args.overwrite)


if __name__ == '__main__':
    main()
