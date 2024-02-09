export CUDA_VISIBLE_DEVICES=$1
python train.py \
--model_name sacrw \
--dataset movi \
--task od \
--monitor avg_ari_fg \
--batch_size 128 \
--split_name "E" \
--num_slots 11 \
--seed 88 \
--slot_size 384 \
--mlp_hidden_size 384 \
--alpha 0 \
--beta 100 \
--additional_position True