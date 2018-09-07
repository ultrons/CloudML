export BUCKET=gcp-demo-acme
export REGION=us-central1
export DATASET=catsNDogs
export TFVERSION=1.8

export train_data="gs://${BUCKET}/${DATASET}/tfrecords_n/train*"
export val_data="gs://${BUCKET}/${DATASET}/tfrecords_n/valid*"
export OUTDIR=gs://${BUCKET}/models_v7/sharkID/sharkID_trained_acc_cfg_12
gsutil -m rm -rf $OUTDIR 
JOBNAME=sharkID_$(date -u +%y%m%d_%H%M%S)
gcloud ml-engine jobs submit training $JOBNAME    \
  --region=$REGION    \
  --module-name=trainer.task    \
  --package-path=${PWD}/sharkID/trainer    \
  --job-dir=$OUTDIR    \
  --staging-bucket=gs://$BUCKET    \
  --scale-tier=BASIC_GPU   \
  --runtime-version=$TFVERSION    \
  --    \
  --train_data_paths=$train_data \
  --eval_data_paths=$val_data \
  --output_dir=$OUTDIR    \
  --pos_weight=1.6165 \
  --hidden_units='32 32' \
  --train_batch_size=32 \
  --dropout_rate=0.47334 \
  --learning_rate=0.00999333 \
  --min_eval_frequency=30 \
  --train_steps=4000

export train_data="gs://${BUCKET}/${DATASET}/tfrecords_no_ovr_sampling/train*"
export val_data="gs://${BUCKET}/${DATASET}/tfrecords_no_ovr_sampling/valid*"
export OUTDIR=gs://${BUCKET}/models_v7/sharkID/sharkID_trained_no_ovr_acc_cfg_19
gsutil -m rm -rf $OUTDIR 
JOBNAME=sharkID_$(date -u +%y%m%d_%H%M%S)
gcloud ml-engine jobs submit training $JOBNAME    \
  --region=$REGION    \
  --module-name=trainer.task    \
  --package-path=${PWD}/sharkID/trainer    \
  --job-dir=$OUTDIR    \
  --staging-bucket=gs://$BUCKET    \
  --scale-tier=BASIC_GPU   \
  --runtime-version=$TFVERSION    \
  --    \
  --train_data_paths=$train_data \
  --eval_data_paths=$val_data \
  --output_dir=$OUTDIR    \
  --pos_weight=1.0070 \
  --hidden_units='64' \
  --train_batch_size=32 \
  --dropout_rate=0.306424006 \
  --learning_rate=0.00986519 \
  --min_eval_frequency=30 \
  --train_steps=4000
