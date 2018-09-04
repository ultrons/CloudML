rm -rf /tmp/sharkID_trained
export BUCKET=gcp-demo-acme
export DATASET=catsNDogs
export PYTHONPATH=${PYTHONPATH}:${PWD}/sharkID
python -m trainer.task    \
	--train_data_paths="gs://${BUCKET}/${DATASET}/tfrecords/train*" \
    	--eval_data_paths="gs://${BUCKET}/${DATASET}/tfrecords/valid*" \
    	--output_dir=/tmp/sharkID_trained  \
      	--train_steps=1 --job-dir=./tmp


