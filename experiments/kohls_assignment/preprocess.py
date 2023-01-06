import argparse
import utils

def main(config):
    df = utils.read_csv(config["data_path"])
    X, y = utils.separate_target_and_features(df)

    X_train, X_test, y_train, y_test = utils.split_train_and_test(
        X, y, train_size=config["train_size"], random_state=config["random_state"])

    imputer = utils.fit_impute(X_train, config)
    x_numeric_imputed = utils.impute_numeric_features(imputer, X_train, config)

    scaler = utils.fit_scalar(x_numeric_imputed, config)
    x_numeric_imputed_scaled = utils.scale_numeric_features(
        scaler, x_numeric_imputed, config)
    
    encoders = utils.fit_ohencoders(X_train, config)
    encoded_features = utils.ohencode_features(encoders, X_train, config)

    X_train_categorical = utils.concat_dfs(encoded_features)
    X_train_final = utils.concat_dfs(
        [X_train_categorical, x_numeric_imputed_scaled])

    X_train_final.to_csv(config["X_train_filepath"], index=False)
    y_train.to_csv(config["y_train_filepath"], index=False)
    X_test.to_csv(config["X_test_filepath"],index=False)
    y_test.to_csv(config["y_test_filepath"],index=False)

    utils.save_transformers(imputer,"encoders/imputer.pkl")
    utils.save_transformers(scaler,"encoders/scaler.pkl")
    for encoder,col in zip(encoders,config["categorical_features"]):
        utils.save_transformers(encoder,"encoders/"+col+"_encoder.pkl")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, required=True,
                        help=' Path to config file with parameters')

    args = parser.parse_args()
    config = utils.read_json(args.config)
    main(config)