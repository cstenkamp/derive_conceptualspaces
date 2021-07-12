import os
from os.path import join
from types import ModuleType

import src.static.classifier_config
from src.static.classifier_config import CLASSIFIER_CHECKPOINT_PATH, SUMMARY_PATH
import subprocess

def write_config(dest_path, config):
    with open(join(dest_path, 'config'), 'w') as configfile:
        configfile.write(config)


def read_config(dest_path):
    try:
        with open(join(dest_path, 'config'), 'r') as configfile:
            return configfile.read()
    except FileNotFoundError:
        return ''


def get_all_debug_confs():
    return {i:eval('main.config.'+i) for i in dir(main.config) if i.startswith('DEBUG')}


def get_all_configs():
    config = {i: eval('src.static.classifier_config.' + i) for i in dir(src.static.classifier_config) if not i.startswith('_')}
    config = {key: val for key, val in config.items() if
              not callable(val) and not isinstance(val, ModuleType) and key.isupper()}
    config_str = '\n'.join(f'{key}: {val}' for key, val in config.items())
    return config_str



def clear_checkpoints_summary():
    for dir in [CLASSIFIER_CHECKPOINT_PATH, SUMMARY_PATH]:
        if os.path.isdir(dir):
            for f in [f for f in os.listdir(dir) if f.endswith(".bak")]:
                os.remove(os.path.join(dir, f))


def check_config(path):
    if read_config(path) != get_all_configs():
        print('The config of the last run differs from this one!')
        oldconf = {key: val for key, val in [(i.split(':')[0], ':'.join(i.split(':')[1:])) for i in read_config(path).split("\n")]}
        newconf = {key: val for key, val in [(i.split(':')[0], ':'.join(i.split(':')[1:])) for i in get_all_configs().split("\n")]}
        print('\n  '.join([f'{key}: {val} in old, {newconf.get(key)} in new' for key, val in oldconf.items() if val != newconf.get(key)]))
        while True:
            answer = input('Overwrite the last checkpoint? [y]es or [n]o (=try to continue this checkpoint) ').lower()
            if answer in ['y', 'n']:
                break
        if answer == 'y':
            clear_checkpoints_summary()
        write_config(path, get_all_configs())



def get_ann_configs():
    lines = []
    with open(src.static.classifier_config.__file__, 'r') as f:
        ann_sets = False
        for line in f.readlines():
            if ann_sets:
                if line.startswith('###'):
                    ann_sets = False
                elif not line.startswith('#') and line.strip() != '':
                    lines.append(line.strip())
            elif 'ANN Settings' in line:
                ann_sets = True
    return lines


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()