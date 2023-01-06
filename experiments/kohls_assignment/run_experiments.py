import preprocess
import train
import utils
import argparse
import glob
import os
import mlflow
import mlflow.sklearn

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, required=True,
                        help=' Path to config file with parameters')
    
    args = parser.parse_args()
    config = utils.read_json(args.config)
    
    with mlflow.start_run():
        
        mlflow.log_dict(config,"config.json")
        
        preprocess.main(config)
        
        for encoder in glob.glob("encoders/*"):
            mlflow.log_artifact(encoder)
        
        mlflow.log_artifact(config["X_train_filepath"])
        mlflow.log_artifact(config["y_train_filepath"])
        mlflow.log_artifact(config["X_test_filepath"])
        mlflow.log_artifact(config["y_test_filepath"])
        model = train.main(config)
        
        X_test = utils.read_csv(config["X_test_filepath"])
        y_test = utils.read_csv(config["y_test_filepath"])

        imputer = utils.load_transformers(os.path.join(config["encoders_dir"],"imputer.pkl"))
        scaler = utils.load_transformers(os.path.join(config["encoders_dir"],"scaler.pkl"))
        ohencoders = []
        for col in config["categorical_features"]:
            ohencoders.append(utils.load_transformers(os.path.join(config["encoders_dir"],col+"_encoder.pkl")))
        
        x_numeric_imputed = utils.impute_numeric_features(imputer, X_test, config)
        x_numeric_imputed_scaled = utils.scale_numeric_features(
        scaler, x_numeric_imputed, config)

        encoded_features = utils.ohencode_features(ohencoders, X_test, config)
        X_test_categorical = utils.concat_dfs(encoded_features)
        X_test_final = utils.concat_dfs(
            [X_test_categorical, x_numeric_imputed_scaled])

        confusion_matrix = utils.get_conf_matrix(model,X_test_final,y_test)
        conf_matrix_fig = utils.confusion_matrix_fig(confusion_matrix)
        mlflow.log_figure(conf_matrix_fig,"conf_matrix_fig.png")

        auc_roc = utils.get_roc_auc(model,X_test_final,y_test)
        mlflow.log_metric("AUC",auc_roc)
        

        model_accuracy = model.score(X_test_final,y_test)
        mlflow.log_metric("mean_accuracy",model_accuracy)


        mlflow.sklearn.log_model(model, "model")

