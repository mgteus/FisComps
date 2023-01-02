import sys
import pandas as pd
import json
import sys

path_ = r'D:\FirefoxDownloads\trainingAndTest'


def main():

    train_path = r'D:\FirefoxDownloads\trainingAndTest\training.json'
    
    with open(train_path, 'r') as file:
        lines = [lin.replace('\n', '') for lin in file.readlines()]

    n_traning_samples = int(lines[0])

    aux_dict = {}
    for lin in lines[1:5]:
        info = json.loads(lin)
        serial = info['serial']
        del info['serial']
        aux_dict[serial] =  info

    


    return lines[:10], n_traning_samples, aux_dict






if __name__ == '__main__':
    print(main())