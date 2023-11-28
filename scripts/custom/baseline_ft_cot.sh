TARGETS=("lrf_8_shot" "multiwd_8_shot")
#MODELS=("flan_t5_large" "flan_t5_xl" "gpt2_medium" "gpt2_large" "t5_large" "t5_3b")
MODELS=("flan_t5_base" "flan_t5_small")
DEVICES="0"


for MODEL in ${MODELS[@]}; do
  for TARGET in ${TARGETS[@]}; do
    python custom_train.py --dataset_key $TARGET --model_key $MODEL --train_key "ft_cot" --devices $DEVICES --batch_size 8 --inference_batch_size 32 --precision 16 --preset_key "ft_cot_t70_8aug"
  done
done


TARGETS=("lrf_16_shot" "multiwd_16_shot")

for MODEL in ${MODELS[@]}; do
  for TARGET in ${TARGETS[@]}; do
    python custom_train.py --dataset_key $TARGET --model_key $MODEL --train_key "ft_cot" --devices $DEVICES --batch_size 8 --inference_batch_size 32 --precision 16 --preset_key "ft_cot_t70_16aug"
  done
done