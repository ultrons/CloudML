
# coding: utf-8

# # General Project Setup

# In[2]:


import os
PROJECT = 'cs231n-vaibhavs' 
REGION = 'us-central1' # Choose an available region for Cloud MLE from https://cloud.google.com/ml-engine/docs/regions.
BUCKET = 'gcp-demo-acme' # REPLACE WITH YOUR BUCKET NAME. Use a regional bucket in the region you selected.


# In[3]:


# for bash
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.8'


# In[1]:


get_ipython().run_line_magic('bash', '')
gcloud config set project $PROJECT
gcloud config set compute/region $REGION


# # Pre-Process Images (Convert to TF Records using DataFlow)

# In[ ]:


get_ipython().run_line_magic('bash', '')
export BUCKET=gcp-demo-acme
gs://gcp-demo-acme/dataset/train_gs.csv
gs://gcp-demo-acme/dataset/valid_gs.csv
echo "SHARK" > /tmp/labels.txt
echo "NO_SHARK" >> /tmp/labels.txt

python -m jpeg_to_tf_record.py        --train_csv gs://${BUCKET}/dataset/train_gs.csv        --validation_csv gs://${BUCKET}/dataset/valid_gs.csv        --labels_file /tmp/labels.txt        --project_id ${PROJECT}        --output_dir gs://${BUCKET}/dataset_dataflow


# # Authorizing Cloud ML access to the Cloud Storage Bucket

# In[4]:


get_ipython().run_line_magic('bash', '')
curl -X GET -H "Content-Type: application/json"     -H "Authorization: Bearer $AUTH_TOKEN"     https://ml.googleapis.com/v1/projects/${PROJECT_ID}:getConfig     | python -c "import json; import sys; response = json.load(sys.stdin);     print response"


# In[5]:


get_ipython().run_line_magic('bash', '')
PROJECT_ID=$PROJECT
AUTH_TOKEN=$(gcloud auth print-access-token)
SVC_ACCOUNT=$(curl -X GET -H "Content-Type: application/json"     -H "Authorization: Bearer $AUTH_TOKEN"     https://ml.googleapis.com/v1/projects/${PROJECT_ID}:getConfig     | python -c "import json; import sys; response = json.load(sys.stdin);     print response['serviceAccount']")

echo "Authorizing the Cloud ML Service account $SVC_ACCOUNT to access files in $BUCKET"
gsutil -m defacl ch -u $SVC_ACCOUNT:R gs://$BUCKET
gsutil -m acl ch -u $SVC_ACCOUNT:R -r gs://$BUCKET  # error message (if bucket is empty) can be ignored
gsutil -m acl ch -u $SVC_ACCOUNT:W gs://$BUCKET


# # Test Local Execution

# In[4]:


get_ipython().run_line_magic('bash', '')
# Local execution
pip install -q tensorflow-hub
rm -rf sharkID_trained
export BUCKET=gcp-demo-acme
export PYTHONPATH=${PYTHONPATH}:${PWD}/sharkID
python -m trainer.task    --train_data_paths="gs://${BUCKET}/dataset/train/*.tfrecord"    --eval_data_paths="gs://${BUCKET}/dataset/valid/*.tfrecord"     --output_dir=${PWD}/sharkID_trained    --train_steps=10 --job-dir=./tmp


# # Tuning HyperParameters

# ## 1. Create parameter search specification
# 
# 
# Experiment-1: Metric -> Accuracy

# In[8]:


get_ipython().run_line_magic('writefile', 'hyperparam_accuracy.yaml')
trainingInput:
  scaleTier: BASIC_GPU
  hyperparameters:
    goal: MINIMIZE
    maxTrials: 30
    maxParallelTrials: 4
    hyperparameterMetricTag: recall
    params:
    - parameterName: hidden_units
      type: CATEGORICAL
      categoricalValues: ['64', '32 32']
    - parameterName: learning_rate
      type: DOUBLE
      scaleType: UNIT_REVERSE_LOG_SCALE
      minValue: 0.0001
      maxValue: 0.01    
    - parameterName: pos_weight
      type: DOUBLE
      scaleType: UNIT_LINEAR_SCALE
      minValue: 1.1
      maxValue: 1.8
    - parameterName: dropout_rate
      type: DOUBLE
      scaleType: UNIT_LINEAR_SCALE
      minValue: 0.3
      maxValue: 0.6
    - parameterName: train_batch_size
      type: DISCRETE
      discreteValues:
      - 8
      - 16
      - 32
        


# In[9]:


get_ipython().run_line_magic('bash', '')
export BUCKET=gcp-demo-acme
export REGION=us-central1
OUTDIR=gs://${BUCKET}/sharkID/hp_tuning_accuracy/sharkID_trained
JOBNAME=sharkID_$(date -u +%y%m%d_%H%M%S)
export TFVERSION=1.8
train_path="gs://${BUCKET}/dataset_split_85_15/train/*.tfrecord"
valid_path="gs://${BUCKET}/dataset_split_85_15/valid/*.tfrecord"
echo $OUTDIR $REGION $JOBNAME

gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME    --region=$REGION    --module-name=trainer.task    --package-path=${PWD}/sharkID/trainer    --job-dir=$OUTDIR    --staging-bucket=gs://$BUCKET    --scale-tier=BASIC_GPU   --runtime-version=$TFVERSION    --config=hyperparam_accuracy.yaml    --    --train_data_paths=${train_path}    --eval_data_paths=${valid_path}     --output_dir=$OUTDIR    --train_steps=1600


# In[11]:


get_ipython().run_line_magic('bash', '')
gsutil ls gs://${BUCKET}/dataset_split_85_15/valid/*.tfrecord


# 
# 
# 
# Experiment-2: Metric -> Recall

# In[6]:


get_ipython().run_line_magic('writefile', 'hyperparam_recall.yaml')
trainingInput:
  scaleTier: BASIC_GPU
  hyperparameters:
    goal: MINIMIZE
    maxTrials: 30
    maxParallelTrials: 4
    hyperparameterMetricTag: accuracy
    params:
    - parameterName: hidden_units
      type: CATEGORICAL
      categoricalValues: ['64', '32 32']
    - parameterName: learning_rate
      type: DOUBLE
      scaleType: UNIT_REVERSE_LOG_SCALE
      minValue: 0.0001
      maxValue: 0.01    
    - parameterName: pos_weight
      type: DOUBLE
      scaleType: UNIT_LINEAR_SCALE
      minValue: 1.1
      maxValue: 1.8
    - parameterName: dropout_rate
      type: DOUBLE
      scaleType: UNIT_LINEAR_SCALE
      minValue: 0.3
      maxValue: 0.6
    - parameterName: train_batch_size
      type: DISCRETE
      discreteValues:
      - 8
      - 16
      - 32
        


# In[7]:


get_ipython().run_line_magic('bash', '')
export BUCKET=gcp-demo-acme
export REGION=us-central1
OUTDIR=gs://${BUCKET}/sharkID/hp_tuning_recall/sharkID_trained
JOBNAME=sharkID_$(date -u +%y%m%d_%H%M%S)
export TFVERSION=1.8
train_path="gs://${BUCKET}/dataset_split_85_15/train/*.tfrecord"
valid_path="gs://${BUCKET}/dataset_split_85_15/valid/*.tfrecord"
echo $OUTDIR $REGION $JOBNAME

gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME    --region=$REGION    --module-name=trainer.task    --package-path=${PWD}/sharkID/trainer    --job-dir=$OUTDIR    --staging-bucket=gs://$BUCKET    --scale-tier=BASIC_GPU   --runtime-version=$TFVERSION    --config=hyperparam_recall.yaml    --    --train_data_paths=${train_path}    --eval_data_paths=${valid_path}     --output_dir=$OUTDIR    --train_steps=1600


# # Train the Model with Best HyperParameter Setting

# In[5]:


get_ipython().run_line_magic('bash', '')
# Submit Training on CloudML
export BUCKET=gcp-demo-acme
export REGION=us-central1
OUTDIR=gs://${BUCKET}/sharkID/sharkID_trained
JOBNAME=sharkID_$(date -u +%y%m%d_%H%M%S)
export TFVERSION=1.8
echo $OUTDIR $REGION $JOBNAME

gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME    --region=$REGION    --module-name=trainer.task    --package-path=${PWD}/sharkID/trainer    --job-dir=$OUTDIR    --staging-bucket=gs://$BUCKET    --scale-tier=BASIC_TPU   --runtime-version=$TFVERSION    --    --train_data_paths="gs://${BUCKET}/dataset/train/*.tfrecord"    --eval_data_paths="gs://${BUCKET}/dataset/valid/*.tfrecord"    --output_dir=$OUTDIR    --train_steps=5000


# # Deploy the Model

# In[37]:


get_ipython().run_line_magic('bash', '')
MODEL_NAME=sharkID
MODEL_VERSION=v1
BUCKET=gcp-demo-acme
REGION='us-west1'
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/sharkID/sharkID_trained/export/exporter | tail -1)
#echo "Run these commands one-by-one (the very first time, you'll create a model and then create a version)"
#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version '1.8'


# # Test Prediction

# In[19]:


get_ipython().run_line_magic('bash', '')
# Checking prediction  
time gcloud ml-engine predict --model=sharkID --version=v3 --json-instances=../test.json


# # Examine Output Through Tensor-Board

# In[12]:


from google.datalab.ml import TensorBoard
TensorBoard().start('gs://'+BUCKET+'/sharkID')

