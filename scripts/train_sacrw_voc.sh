export CUDA_VISIBLE_DEVICES=$1
python train.py \
--model_name sacrw \
--dataset voc \
--task od \
--monitor avg_ari_fg \
--batch_size 128 \
--num_slots 4 \
--seed 42 \
--slot_size 384 \
--mlp_hidden_size 384 \
--alpha 1 \
--beta 100