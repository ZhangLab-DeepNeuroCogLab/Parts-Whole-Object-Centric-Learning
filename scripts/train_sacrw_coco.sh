export CUDA_VISIBLE_DEVICES=$1
python train.py \
--model_name sacrw \
--dataset coco \
--task od \
--monitor avg_ari_fg \
--batch_size 128 \
--num_slots 7 \
--seed 42 \
--slot_size 384 \
--mlp_hidden_size 384 \
--alpha 0.1 \
--beta 100