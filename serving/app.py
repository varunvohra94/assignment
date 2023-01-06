from regex import P
import utils
from flask import Flask,request
import pandas as pd
import os

app = Flask(__name__)
print("Hello")
model = utils.load_model("models/model.pkl")
print(model)
config = utils.read_json("config.json")
print(config)
imputer = utils.load_transformers(os.path.join(config["encoders_dir"],"imputer.pkl"))
print(imputer)
scaler = utils.load_transformers(os.path.join(config["encoders_dir"],"scaler.pkl"))
print(scaler)
ohencoders = []
for col in config["categorical_features"]:
    ohencoders.append(utils.load_transformers(os.path.join(config["encoders_dir"],col+"_encoder.pkl")))
print(ohencoders)

@app.route('/predict',methods=['POST'])
def predict():
    print("Enter predict")
    req = request.json.get('instances')
    print(req)
    df = pd.DataFrame([req],columns = ["x1","x2","x3","x4","x5","x6","x7"])
    x_numeric_imputed = utils.impute_numeric_features(imputer, df, config)
    x_numeric_imputed_scaled = utils.scale_numeric_features(
    scaler, x_numeric_imputed, config)

    encoded_features = utils.ohencode_features(ohencoders, df, config)
    X_test_categorical = utils.concat_dfs(encoded_features)
    X_test_final = utils.concat_dfs(
        [X_test_categorical, x_numeric_imputed_scaled])
    
    result = model.predict(X_test_final)
    result = {"prediction":str(result[0])}
    return result

@app.route('/healthz')
def healthz():
    return "OK"


if __name__=='__main__':
    app.run(host="0.0.0.0",port=3000)