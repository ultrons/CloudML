MODEL_NAME=sharkID
MODEL_VERSION=v6
BUCKET=gcp-demo-acme
REGION='us-west1'
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/models_v3/sharkID/sharkID_trained_dgraph_w1/export/exporter | tail -1)
#echo "Run these commands one-by-one (the very first time, you'll create a model and then create a version)"
gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
#gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version '1.8'

