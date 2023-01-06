import argparse
import utils
import os

def main(config):
    X_train = utils.read_csv(config["X_train_filepath"])    
    y_train = utils.read_csv(config["y_train_filepath"])

    
    model = utils.train_model(X_train,y_train,config)
    utils.save_model(model,config)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, required=True,
                        help=' Path to config file with parameters')

    args = parser.parse_args()
    config = utils.read_json(args.config)
    main(config)