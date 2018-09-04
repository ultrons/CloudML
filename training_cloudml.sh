export BUCKET=gcp-demo-acme
export REGION=us-central1
export DATASET=catsNDogs
export OUTDIR=gs://${BUCKET}/models_v3/sharkID/sharkID_trained_dgraph_w1
JOBNAME=sharkID_$(date -u +%y%m%d_%H%M%S)
export TFVERSION=1.8
echo $OUTDIR $REGION $JOBNAME

gsutil -m rm -rf $OUTDIR 
gcloud ml-engine jobs submit training $JOBNAME    \
  --region=$REGION    \
  --module-name=trainer.task    \
  --package-path=${PWD}/sharkID/trainer    \
  --job-dir=$OUTDIR    \
  --staging-bucket=gs://$BUCKET    \
  --scale-tier=BASIC_GPU   \
  --runtime-version=$TFVERSION    \
  --    \
  --train_data_paths="gs://${BUCKET}/${DATASET}/tfrecords_n/train*" \
  --eval_data_paths="gs://${BUCKET}/${DATASET}/tfrecords_n/valid*" \
  --output_dir=$OUTDIR    \
  --pos_weight=1.0 \
  --hidden_units='64' \
  --train_batch_size=16 \
  --dropout_rate=0.3035 \
  --learning_rate=0.0001932625 \
  --min_eval_frequency=30 \
  --train_steps=15000

