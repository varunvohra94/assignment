cp -r ../experiments/kohls_assignment/*.py .
cp -r ../experiments/kohls_assignment/*.json .
mkdir encoders
cp -r ../experiments/kohls_assignment/encoders/*.pkl ./encoders
mkdir models
cp -r ../experiments/kohls_assignment/models/*.pkl ./models/
docker build --no-cache=true -t prediction_container .