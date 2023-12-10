export CUDA_VISIBLE_DEVICES=$1
python test.py \
--model_name sacrw \
--dataset cars \
--task fe \
--monitor avg_iou \
--batch_size 128 \
--num_slots 2 \
--seed 42 \
--slot_size 384 \
--mlp_hidden_size 384