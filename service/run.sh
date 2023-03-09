gdown --id "${PRODUCTS_DATASET_URL}"
python create_dataset.py
python dataset.py
python finetuning.py \
                        --output_dir "${OUTPUT_FOLDER}" \
                        --model_name_or_path "${REPO_ID}" \
                        --train_file "${OUT_JSON}" \
                        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
                        --per_device_train_batch_size "${BATCH_SIZE}" \
                        --max_seq_length 77 \
                        --image_column 'image' \
                        --caption_column caption \
                        --learning_rate  0.006737947 \
                        --warmup_steps '0' \
                        --weight_decay 0.1\
                        --remove_unused_columns false \
                        --do_train \
                        --overwrite_output_dir true