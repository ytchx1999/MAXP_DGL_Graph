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
--load-embs \
--load-label-emb \
--seed 0 \
--num-runs 1 \
--threshold 0.9 \
--epoch-setting 500 200 200 \
--lr 0.001 \
--batch-size 50000 \
--num-hidden 512 \
--dropout 0.5 \
--attn-drop 0.4 \
--input-drop 0.2 \
--label-drop 0.5 \
--K 5 \
--label-K 9 \