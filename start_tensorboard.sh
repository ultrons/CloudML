export BUCKET=gcp-demo-acme


#tensorboard --port 3007 --logdir==/home/singh_vaibhav_83/catsNDogs/models/hp_tuning_acc/sharkID_trained &
#tensorboard --port 3008 --logdir==/home/singh_vaibhav_83/catsNDogs/models/hp_tuning_precision/sharkID_trained &
#tensorboard --port 3009 --logdir==/home/singh_vaibhav_83/catsNDogs/models/hp_tuning_recall/sharkID_trained &

tensorboard --port=3007 --logdir==/home/singh_vaibhav_83/catsNDogs/models/hp_tuning_acc/sharkID_trained &
tensorboard --port=3008 --logdir==/home/singh_vaibhav_83/catsNDogs/models/hp_tuning_precision/sharkID_trained &
tensorboard --port=3009 --logdir==/home/singh_vaibhav_83/catsNDogs/models/hp_tuning_recall/sharkID_trained &

/home/singh_vaibhav_83/catsNDogs/models
#tensorboard --port 3007 --logdir==/home/singh_vaibhav_83/catsNDogs/models/hp_tuning_acc/sharkID_trained
#tensorboard --port 3007 --logdir==/home/singh_vaibhav_83/catsNDogs/models/hp_tuning_recall/sharkID_trained
#tensorboard --port 3007 --logdir==/home/singh_vaibhav_83/catsNDogs/models/hp_tuning_precision/sharkID_trained
