#!/usr/bin/env bash
set -euxo pipefail
#time HADOOP_CONF_DIR=/etc/hadoop/conf ./bin/flink run -c com.google.cloud.flink.sandbox.BigShuffle -p 396 -m yarn-cluster -ys 4 -yn 99 -ytm 12000 ~/flink-sandbox-assembly-0.1.0-SNAPSHOT.jar 'gs://sidhom-scratch-us-central1/teragen/100tb/ascii_sort_1GB_input.0000*' gs://sidhom-scratch-us-central1/bigshuffle/result
#time gcloud dataproc clusters create sidhom-flink --master-machine-type n1-standard-4 --worker-machine-type n1-standard-4 --num-workers 100 --initialization-actions=gs://dataproc-initialization-actions/flink/flink.sh --metadata=flink-start-yarn-session=false --zone us-central1-b
#time gcloud dataproc clusters create goenka-flink-155 --num-workers=2 --initialization-actions gs://clouddfe-goenka/flink/flink-init-155.sh,gs://clouddfe-goenka/flink/docker-init.sh
#time gcloud compute instances list  | grep goenka-flink-155-16 | sed "s/ .*//" | xargs -I INSTANCE -P 100 gcloud compute ssh yarn@INSTANCE --command="docker pull gcr.io/dataflow-build/goenka/beam_fnapi_python:latest"
#time /usr/lib/flink/bin/yarn-session.sh -n 2 -tm 10240 -s 4 -d -nm flink_yarn

#readonly FLINK_VERSION=1.5.5
readonly FLINK_VERSION=$(/usr/share/google/get_metadata_value attributes/flink_version)
readonly FLINK_TOPLEVEL="flink-$FLINK_VERSION"
#readonly FLINK_GCS="gs://clouddfe-goenka/flink/dist/flink-$FLINK_VERSION-bin-hadoop28-scala_2.11.tgz"
readonly FLINK_GCS=$(/usr/share/google/get_metadata_value attributes/work_dir)/flink-$FLINK_VERSION-bin-hadoop28-scala_2.11.tgz
readonly FLINK_LOCAL="/tmp/flink-$FLINK_VERSION-bin-hadoop28-scala_2.11.tgz"
readonly FLINK_INSTALL_DIR='/usr/lib/flink'
function is_master() {
  local role="$(/usr/share/google/get_metadata_value attributes/dataproc-role)"
  if [[ "$role" == 'Master' ]] ; then
    true
  else
    false
  fi
}
function primary_master() {
  local primary="$(/usr/share/google/get_metadata_value attributes/dataproc-master)"
  echo -n "$primary"
}
function install_flink() {
  gsutil cp "$FLINK_GCS" "$FLINK_LOCAL"
  tar -xvf "$FLINK_LOCAL" -C /tmp
  mv "/tmp/$FLINK_TOPLEVEL" "$FLINK_INSTALL_DIR"
  rm "$FLINK_LOCAL"
}
function configure_flink() {
  local hdfs_master="$(primary_master)"
  # TODO(sidhom): How do we get HDFS port from config?
  local history_dir="hdfs://$hdfs_master:8020/user/yarn/flink-history"
  hdfs dfs -mkdir "$history_dir"
  #hdfs dfs -chown yarn "$history_dir"
  mkdir /var/log/flink
  chmod a+rwxt /var/log/flink
  cat >>"$FLINK_INSTALL_DIR/conf/flink-conf.yaml" <<EOF
historyserver.web.port: 8082
jobmanager.archive.fs.dir: $history_dir
historyserver.archive.fs.dir: $history_dir
historyserver.archive.fs.refresh-interval: 10000
akka.ask.timeout: 60s
restart-strategy: none
env.log.dir: /var/log/flink
EOF
  "$FLINK_INSTALL_DIR/bin/historyserver.sh" start
}
function main() {
  if is_master ; then
    install_flink
    configure_flink
  fi
}
main "$@"
