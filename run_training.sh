# Train
# comment lines within command with #`bool_arg` to avoid breaking

python3 -u src/main.py \
--language "english" \
--mode "train" \
--n_epochs 20 \
--early_stop \
--lr 0.5 \
--lr_decay \
--init_type "zeros" \
--rand_seed 7 \
`#--save_model` \
--models_dir "models/" \
--train_file "data/english/train/wsj_train.conll06" \
--train_slice 0 \
--print_every 500 \