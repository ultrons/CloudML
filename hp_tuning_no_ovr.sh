export BUCKET=gcp-demo-acme
export DATASET=catsNDogs
export REGION=us-central1
export TFVERSION=1.8
export train_data="gs://${BUCKET}/${DATASET}/tfrecords_no_ovr_sampling/train*"
export val_data="gs://${BUCKET}/${DATASET}/tfrecords_no_ovr_sampling/valid*"

export OUTDIR=gs://${BUCKET}/models_no_ovr/sharkID/hp_tuning_acc/sharkID_trained
JOBNAME=sharkID_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME \
    	--region=$REGION    --module-name=trainer.task    \
	--package-path=${PWD}/sharkID/trainer    \
	--job-dir=$OUTDIR    \
	--staging-bucket=gs://$BUCKET    \
	--scale-tier=BASIC_GPU   \
	--runtime-version=$TFVERSION    \
	--config=hyperparam_accuracy.yaml    \
	--    \
	--train_data_paths=$train_data \
    	--eval_data_paths=$val_data \
	--output_dir=$OUTDIR    \
	--train_steps=2000


export OUTDIR=gs://${BUCKET}/models_no_ovr/sharkID/hp_tuning_recall/sharkID_trained
JOBNAME=sharkID_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME \
    	--region=$REGION    --module-name=trainer.task    \
	--package-path=${PWD}/sharkID/trainer    \
	--job-dir=$OUTDIR    \
	--staging-bucket=gs://$BUCKET    \
	--scale-tier=BASIC_GPU   \
	--runtime-version=$TFVERSION    \
	--config=hyperparam_recall.yaml    \
	--    \
	--train_data_paths=$train_data \
    	--eval_data_paths=$val_data \
	--output_dir=$OUTDIR    \
	--train_steps=2000


export OUTDIR=gs://${BUCKET}/models_no_ovr/sharkID/hp_tuning_precision/sharkID_trained
JOBNAME=sharkID_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME \
    	--region=$REGION    --module-name=trainer.task    \
	--package-path=${PWD}/sharkID/trainer    \
	--job-dir=$OUTDIR    \
	--staging-bucket=gs://$BUCKET    \
	--scale-tier=BASIC_GPU   \
	--runtime-version=$TFVERSION    \
	--config=hyperparam_precision.yaml    \
	--    \
	--train_data_paths=$train_data \
    	--eval_data_paths=$val_data \
	--output_dir=$OUTDIR    \
	--train_steps=2000

