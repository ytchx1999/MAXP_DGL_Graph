cd "$(dirname $0)"

python3 ../main.py \
--seed 0 \
--gpu 1 \
--method R_GAMLP \
--stages 101 \
--train-num-epochs 0 \
--threshold 0.85 \
--input-drop 0.1 \
--att-drop 0.2 \
--label-drop 0 \
--pre-process \
--residual \
--dataset maxp \
--eval 5 \
--batch 5000 \
--patience 300 \
--n-layers-1 4 \
--n-layers-2 6 \
--bns \
--gama 0.1 \
--label-num-hops 9 \
--num-hops 9 \
--hidden 1024 \
--temp 0.001 \
--num-runs 10 \
--act leaky_relu \
--all-train

# --flag \
# --use-rlu \
# --method R_GAMLP_RLU \
# --act sigmoid \






