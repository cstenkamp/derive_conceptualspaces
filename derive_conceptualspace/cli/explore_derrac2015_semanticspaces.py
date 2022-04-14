import os
import click
from dotenv import load_dotenv

from derive_conceptualspace.load_data.load_semanticspaces import (
    get_raw_movies_dataset,
    get_raw_places_dataset,
    get_all_goodkappa,
    display_svm as display_svm_base,
    get_all as get_all_base,
    count_raw_places_dataset,
    get_all_places, get_all_movies
)

#TODO: make this usable! No absolute paths in the derive_conceptualspace.load_data.load_semanticspaces file, properly do logging and env-vars and arguments here, ...

@click.group()
@click.option("--log", type=str, default="INFO", help="log-level for logging-module. one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]. Defaults to INFO.")
@click.option("--logfile", type=str, default="", help="logfile to log to. If not set, it will be logged to standard stdout/stderr",)
@click.option("--env-file", type=click.Path(exists=True), help="If you want to provide environment-variables using .env-files you can provide the path to a .env-file here.",)
@click.option("--read-env/--no-read-env", default=True, help="If the program should read a `.env`-file at the base of the repository. Defaults to True",)
@click.pass_context
def main(ctx, log="INFO", logfile=None, env_file=None, read_env=True,):
    """
    This is the main CLI to look at the semanticspaces-datasets as uploaded by Derrac2015 to https://www.cs.cf.ac.uk/semanticspaces/
    """
    ctx.ensure_object(dict)
    if env_file:
        load_dotenv(env_file)
    if read_env and os.path.isfile(".env"):
        print(f"detected an env-file at {os.path.abspath('.env')}, loading it!")
        load_dotenv(".env")

@main.command()
def raw_movies_dataset():
    get_raw_movies_dataset()

@main.command()
def raw_places_dataset():
    get_raw_places_dataset()

@main.command()
def all_goodkappa():
    get_all_goodkappa("places")

@main.command()
def display_svm():
    display_svm_base()

@main.command()
def get_all():
    get_all_base()

@main.command()
def count_places():
    count_raw_places_dataset()

@main.command()
def stats_places():
    get_all_places()

@main.command()
def stats_movies():
    get_all_movies()



if __name__ == "__main__":
    main(obj={})

