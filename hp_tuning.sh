export BUCKET=gcp-demo-acme
export DATASET=catsNDogs
export REGION=us-central1
export OUTDIR=gs://${BUCKET}/models/sharkID/hp_tuning/sharkID_trained
JOBNAME=sharkID_$(date -u +%y%m%d_%H%M%S)
export TFVERSION=1.8
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
	--train_data_paths="gs://${BUCKET}/${DATASET}/tfrecords/train*" \
    	--eval_data_paths="gs://${BUCKET}/${DATASET}/tfrecords/valid*" \
	--output_dir=$OUTDIR    \
	--train_steps=1000
