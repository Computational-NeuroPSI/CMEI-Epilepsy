import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# ---------------------------------------------------
def get_network_config(idx = None, json_file_name = None, sim_info = False):
    if idx == None:
        idx = 0

    if json_file_name == None:
        raise ValueError("Plese, specify the json_file_name containing the model parameters")    
    
    with open(json_file_name, 'r') as file:
        data = json.load(file)

        return data[0]