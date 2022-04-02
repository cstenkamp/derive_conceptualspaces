#!/usr/bin/env python3
import os
import pathlib
import re
from os.path import (
    abspath,
    basename,
    dirname,
    isdir,
    isfile,
    join,
    split,
    splitext,
)

import toml
from setuptools import find_packages, setup

TOML_TAKE = {"name", "description", "repository", "python"}
TOML_CHANGE = {"repository": "url", "python": "python_requires"}

SUPER_FOLDER = None

# TODO find a way to combine pyproject.toml requirements with requriements.txt and this

def get_version(projectname):
    # https://stackoverflow.com/a/7071358/5122790
    VERSIONFILE = f"{projectname.replace('-','_')}/_version.py"
    if SUPER_FOLDER:
        VERSIONFILE = SUPER_FOLDER + "/" + VERSIONFILE
    verstrline = open(VERSIONFILE, "rt").read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


def main():
    setup_data = _parse_project_toml(pathlib.Path("pyproject.toml"), "project")
    package_kwargs = {} if not SUPER_FOLDER else {"where": SUPER_FOLDER[:-1]}
    setup(
        version=get_version(setup_data["name"]),
        package_dir={
            setup_data["name"]: (SUPER_FOLDER + "/" + setup_data["name"]) if SUPER_FOLDER else setup_data["name"]
        },
        packages=find_packages(exclude=["test"], **package_kwargs),
        install_requires=_read_requirements(),
        # tests_require=_read_requirements("requirements-dev.txt"),
        dependency_links=_read_requirements(only_gits=True),
        # TODO test_suit for pytest? https://python-packaging.readthedocs.io/en/latest/testing.html
        # scripts=[os.path.join(setup_data["name"], "cli", i) for i in os.listdir(os.path.join(setup_data["name"], "cli")) if not i.startswith("_")] if os.path.isdir(os.path.join(setup_data["name"], "cli")) else [],
        scripts=[
            join(setup_data["name"], "cli", i)
            for i in os.listdir(join(dirname(__file__), setup_data["name"], "cli"))
            if not i.startswith("_")
        ]
        if isdir(join(dirname(__file__), setup_data["name"], "cli"))
        else [],
        include_package_data=True,
        **setup_data,
    )


def _read_requirements(name="requirements.txt", only_gits=False):
    with open(name, "r") as fh:
        lines = fh.readlines()
        lines = [
            line[: line.find(" #") if line.find(" #") > 0 else None].strip()
            for line in lines
            if not line.strip().startswith("#")
        ]
        if not only_gits:
            lines = [line for line in lines if not line.startswith("git+")]
        else:
            lines = [line for line in lines if line.startswith("git+")]
        return lines


def _parse_project_toml(pyproject_path, app_name):
    # see https://stackoverflow.com/a/62372195
    pyproject_text = pyproject_path.read_text()
    pyproject_data = toml.loads(pyproject_text)
    # pyproject_data['project']['name'] = pkg_resources.safe_name(pyproject_data['project']['name'])
    setup_data = {key: val for key, val in pyproject_data["project"].items() if key in TOML_TAKE}
    setup_data = {TOML_CHANGE.get(key, key): val for key, val in setup_data.items()}
    if "authors" in pyproject_data["project"]:
        try:
            setup_data["author"] = " ".join([i[: i.find("<")].strip() for i in pyproject_data["project"]["authors"]])
            setup_data["author_email"] = " ".join(
                [i[i.find("<") + 1 : i.find(">")].strip() for i in pyproject_data["project"]["authors"]]
            )
        except Exception:
            setup_data["author"] = " ".join(
                [i.get("email") for i in pyproject_data["project"]["authors"] if i.get("email")]
            )
            setup_data["author_email"] = " ".join(
                [i.get("name") for i in pyproject_data["project"]["authors"] if i.get("name")]
            )
    if "readme" in pyproject_data["project"]:
        with open(pyproject_data["project"]["readme"], "r") as fh:
            setup_data["long_description"] = fh.read()
        setup_data["long_description_content_type"] = "text/markdown"
    return setup_data


if __name__ == "__main__":
    main()
