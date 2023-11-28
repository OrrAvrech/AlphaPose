set -x

CONFIG=$1
CKPT=$2
VIDEO=$3
OUTDIR=${4:-"./examples/res"}
ROOTDIR=${5:-"/proj/vondrick2/orr/AlphaPose"}

python ${ROOTDIR}/scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --outdir ${OUTDIR} \
    --detector yolo --only_pose --save_video --vis_fast
