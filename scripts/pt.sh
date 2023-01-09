#!/usr/bin/env bash

# an example to do pre-training: (not that `/path/to/imagenet` should contain directories named `train` and `val`)
# > cd /path/to/SparK
# > bash ./scripts/pt.sh experiment_name /path/to/imagenet --num_nodes=1 --ngpu_per_node=8 --node_rank=0 --master_address=128.0.0.0 --master_port=30000 --model=res50 --ep=400

####### template begins #######
SCRIPTS_DIR=$(cd $(dirname $0); pwd)
cd "${SCRIPTS_DIR}"
cd ../
SPARK_DIR=$(pwd)
echo "SPARK_DIR=${SPARK_DIR}"

shopt -s expand_aliases
alias python=python3
alias to_scripts_dir='cd "${SCRIPTS_DIR}"'
alias to_spark_dir='cd "${SPARK_DIR}"'
alias print='echo "$(date +"[%m-%d %H:%M:%S]") (exp.sh)=>"'
function mkd() {
  mkdir -p "$1" >/dev/null 2>&1
}
####### template ends #######


EXP_NAME=$1
DATA_PATH=$2

EXP_DIR="${SPARK_DIR}/output_${EXP_NAME}"
mkd "${EXP_DIR}"


print "===================== Args ====================="
print "EXP_NAME: ${EXP_NAME}"
print "DATA_PATH: ${DATA_PATH}"
print "EXP_DIR: ${EXP_DIR}"
print "[other_args sent to launch.py]: ${*:3}"
print "================================================"
print ""


print "============== Pretraining starts =============="
to_spark_dir
python launch.py \
--main_py_relpath main.py \
--exp_name "${EXP_NAME}" \
--data_path "${DATA_PATH}" \
--exp_dir "${EXP_DIR}" \
"${*:3}"
print "============== Pretraining ends =============="
