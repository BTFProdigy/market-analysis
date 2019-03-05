import os.path

import configparser


def get_config(file):
    config = configparser.ConfigParser()
    config.read(os.path.dirname(__file__) + '/' + file)

    return config