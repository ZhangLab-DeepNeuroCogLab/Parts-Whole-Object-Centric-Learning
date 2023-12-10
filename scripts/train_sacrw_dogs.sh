export CUDA_VISIBLE_DEVICES=$1
python train.py \
--model_name sacrw \
--dataset dogs \
--task fe \
--monitor avg_iou \
--batch_size 128 \
--num_slots 2 \
--seed 42 \
--slot_size 384 \
--mlp_hidden_size 384