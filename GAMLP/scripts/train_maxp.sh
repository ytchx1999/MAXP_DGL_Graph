cd "$(dirname $0)"

python3 ../main.py \
--seed 0 \
--gpu 0 \
--method R_GAMLP \
--stages 101 \
--train-num-epochs 0 \
--threshold 0.85 \
--input-drop 0.2 \
--att-drop 0.5 \
--label-drop 0 \
--pre-process \
--residual \
--dataset maxp \
--num-runs 1 \
--eval 5 \
--batch 5000 \
--patience 300 \
--n-layers-1 4 \
--n-layers-2 4 \
--bns \
--gama 0.1 \
--label-num-hops 9 \
--num-hops 6 \
--hidden 1024 \
--temp 0.001 \
--num-runs 10 \
--act sigmoid \
--flag \
--all-train


# --use-rlu \
# --method R_GAMLP_RLU \
# --act leaky_relu \





