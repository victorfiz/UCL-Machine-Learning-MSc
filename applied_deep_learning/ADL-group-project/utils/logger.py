import logging
import sys, os
from datetime import datetime

def setup_folder_and_logger(name):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(f'logs/{experiment_name}'):
        os.makedirs(f'logs/{experiment_name}')
    
    experiment_folder = f'logs/{experiment_name}'   
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Log to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Log to file
    file_handler = logging.FileHandler(f'{experiment_folder}/run.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return experiment_folder, logger