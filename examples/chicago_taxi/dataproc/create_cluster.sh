#!/usr/bin/env bash

#set -x
set -e
USER=`whoami`
CLUSTER_NAME="$USER-flink-156"
NUM_WORKERS=2
FLINK_VERSION=1.5.6
WORK_DIR="gs://clouddfe-$USER/tmp"
CLOUD_WORKER_IMAGE="gcr.io/dataflow-build/$USER/beam_fnapi_python:latest"
TASK_MANAGER_MEM=10240
FLINK_LOCAL_PORT=8081
TASK_MANAGER_SLOTS=1
DATAPROC_VERSION=1.2

MASTER_NAME="$CLUSTER_NAME-m"
FLINK_INIT="$WORK_DIR/flink/flink-init-dataproc.sh"
DOCKER_INIT="$WORK_DIR/flink/docker-init.sh"
LOCAL_WORKER_IMAGE="$USER-docker-apache.bintray.io/beam/python:latest"
FLINK_DOWNLOAD_URL="http://archive.apache.org/dist/flink/flink-$FLINK_VERSION/flink-$FLINK_VERSION-bin-hadoop28-scala_2.11.tgz"

YARN_APPLICATION=""
YARN_APPLICATION_MASTER=""


function is_master() {
  local role="$(/usr/share/google/get_metadata_value attributes/dataproc-role)"
  if [[ "$role" == 'Master' ]] ; then
    true
  else
    false
  fi
}

function get_leader() {
  local i=0
  local -A application_ids
  local -A application_masters
  #gcloud compute ssh yarn@$MASTER_NAME --command="yarn application -list" | grep "$CLUSTER_NAME"
  echo "Yarn Applications"
  while read line; do
     echo $line
    application_ids[$i]=`echo $line | sed "s/ .*//"`
    application_masters[$i]=`echo $line | sed "s/.*$CLUSTER_NAME/$CLUSTER_NAME/" | sed "s/ .*//"`
    i=$((i+1))
  done <<< $(gcloud compute ssh yarn@$MASTER_NAME --command="yarn application -list" | grep "$CLUSTER_NAME")

  if [ $i != 1 ]; then
    echo "Multiple applications found. Make sure that only 1 application is running on the cluster."
    for app in ${application_ids[*]};
    do
      echo $app
    done

    echo "Execute 'gcloud compute ssh yarn@$MASTER_NAME --command=\"yarn application -kill <APP_NAME>\"' to kill the yarn application."
    exit 1
  fi

  YARN_APPLICATION=${application_ids[0]}
  YARN_APPLICATION_MASTER=${application_masters[0]}
  echo "Using Yarn Application $YARN_APPLICATION $YARN_APPLICATION_MASTER"
}

function upload_worker_image() {
  echo "Tagging worker image $LOCAL_WORKER_IMAGE to $CLOUD_WORKER_IMAGE"
  docker tag $LOCAL_WORKER_IMAGE $CLOUD_WORKER_IMAGE
  echo "Pushing worker image $CLOUD_WORKER_IMAGE to GCR"
  docker push $CLOUD_WORKER_IMAGE
}

function pull_worker_image() {
  echo "Pulling worker image $CLOUD_WORKER_IMAGE on workers $(gcloud compute instances list | sed "s/ .*//" | grep "^\($CLUSTER_NAME-m$\|$CLUSTER_NAME-w-[a-zA-Z0-9]*$\)")"
  gcloud compute instances list | sed "s/ .*//" | grep "^\($CLUSTER_NAME-m$\|$CLUSTER_NAME-w-[a-zA-Z0-9]*$\)" | xargs -I INSTANCE -P 100 gcloud compute ssh yarn@INSTANCE --command="docker pull $CLOUD_WORKER_IMAGE"
}

function start_yarn_application() {
  echo "Starting yarn application on $MASTER_NAME"
  execute_on_master "/usr/lib/flink/bin/yarn-session.sh -n $NUM_WORKERS -tm $TASK_MANAGER_MEM -s $TASK_MANAGER_SLOTS -d -nm flink_yarn"
}

function execute_on_master() {
  gcloud compute ssh yarn@$MASTER_NAME --command="$1"
}

function upload_resources() {
  local TMP_FOLDER=`mktemp -d -t flink_tmp_XXXX`

  echo "Downloading flink at $TMP_FOLDER"
  wget -P $TMP_FOLDER $FLINK_DOWNLOAD_URL

  echo "Uploading resources to GCS $WORK_DIR"
  cp ./create_cluster.sh $TMP_FOLDER
  cp ./docker-init.sh $TMP_FOLDER
  cp ./flink-init-dataproc.sh $TMP_FOLDER

  gsutil cp -r $TMP_FOLDER/* $WORK_DIR/flink

  rm -r $TMP_FOLDER
}

function start_tunnel() {
  local job_server_config=`execute_on_master "curl -s \"http://$YARN_APPLICATION_MASTER/jobmanager/config\""`
  local key="jobmanager.rpc.port"
  local yarn_application_master_host=`echo $YARN_APPLICATION_MASTER | cut -d ":" -f1`

  jobmanager_rpc_port=`echo $job_server_config | python -c "import sys, json; print [ e['value'] for e in json.load(sys.stdin) if e['key'] == u'$key'][0]"`
  local tunnel_command="gcloud compute ssh $MASTER_NAME -- -L $FLINK_LOCAL_PORT:$YARN_APPLICATION_MASTER -L $jobmanager_rpc_port:$yarn_application_master_host:$jobmanager_rpc_port -D 1080"
  local kill_command="gcloud compute ssh yarn@$MASTER_NAME --command=\"yarn application -kill $YARN_APPLICATION\""
  echo "===================Closing the shell does not stop the yarn application==================="
  echo "Execute \"$kill_command\" to kill the yarn application."
  echo "Starting tunnel \"$tunnel_command\""
  echo "Exposing flink jobserver at localhost:$FLINK_LOCAL_PORT"
  gcloud compute ssh yarn@$MASTER_NAME -- -L $FLINK_LOCAL_PORT:$YARN_APPLICATION_MASTER -L $jobmanager_rpc_port:$yarn_application_master_host:$jobmanager_rpc_port -D 1080
  echo "===================Closing the shell does not stop the yarn application==================="
  echo "Execute \"$kill_command\" to kill the yarn application."
  echo "To re-establish tunnel use \"$tunnel_command\""
}

function create_cluster() {
  echo "Starting dataproc cluster."
  gcloud dataproc clusters create $CLUSTER_NAME --num-workers=$NUM_WORKERS --initialization-actions $FLINK_INIT,$DOCKER_INIT --metadata flink_version=$FLINK_VERSION,work_dir=$WORK_DIR/flink --image-version=$DATAPROC_VERSION
  echo "Sleeping for 30 sec"
  sleep 30s
}

function main() {
  upload_resources
  create_cluster # Comment this line to use existing cluster.
  start_yarn_application # Comment this line if yarn application is already running on the cluster.
  get_leader
  upload_worker_image
  pull_worker_image
  start_tunnel
}

main "$@"