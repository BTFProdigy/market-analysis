from os.path import dirname

MAIN_DIRECTORY = dirname(__file__)

def get_models_path():
    # return MAIN_DIRECTORY+"/neural_net/model/"
    return MAIN_DIRECTORY + "/neural_net/model_updated/"
    # return MAIN_DIRECTORY+"/neural_net/model/new_models"

def get_log_dir():
    return MAIN_DIRECTORY + '/neural_net/logs/'

def get_scalars_path():
    return MAIN_DIRECTORY + '/environment/scalers/'

def get_behavior_path():
    return MAIN_DIRECTORY + '/'