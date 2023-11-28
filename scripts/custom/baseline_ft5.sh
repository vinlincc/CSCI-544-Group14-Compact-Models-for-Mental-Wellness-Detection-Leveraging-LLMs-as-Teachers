TARGETS=("lrf" "multiwd")
#MODELS=("flan_t5_large" "flan_t5_xl" "gpt2_medium" "gpt2_large" "t5_large" "t5_3b")
#MODELS=("gpt2" "flan_t5_large" "flan_t5_xl" "gpt2_medium" "gpt2_large" "t5_large" "t5_3b")
MODELS=("flan_t5_base" "flan_t5_small" "t5_base" "t5_small")
DEVICES="0"


for MODEL in ${MODELS[@]}; do
  for TARGET in ${TARGETS[@]}; do
    python custom_train.py --dataset_key $TARGET --model_key $MODEL --train_key "ft" --preset_key "ft" --devices $DEVICES --batch_size 8 --inference_batch_size 32 --precision 16
  done
done

