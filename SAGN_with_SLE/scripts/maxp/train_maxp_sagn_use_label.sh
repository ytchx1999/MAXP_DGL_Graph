cd "$(dirname $0)"
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
--dataset maxp \
--gpu $gpu \
--aggr-gpu -1 \
--eval-every 1 \
--model sagn \
--zero-inits \
--chunks 1 \
--memory-efficient \
--seed 0 \
--num-runs 1 \
--threshold 0.7 \
--epoch-setting 100 100 100 \
--lr 0.001 \
--batch-size 5000 \
--num-hidden 1024 \
--dropout 0.5 \
--attn-drop 0.4 \
--input-drop 0.2 \
--label-drop 0.5 \
--K 5 \
--all-train \
--label-K 14 \
--use-labels \
# --load-embs \
# --load-label-emb \
