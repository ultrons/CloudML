results=`gcloud ml-engine predict --model=sharkID --version=v6 --json-instances=$1`
echo "Prediction for $1: $results"

