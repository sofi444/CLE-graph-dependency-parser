# Train
# comment lines within command with #`bool_arg` to avoid breaking

python3 src/main.py \
--language "english" \
--mode "train" \
--n_epochs 3 \
`#--early_stop` \
--lr 0.3 \
`#--lr_decay` \
--init_type "zeros" \
--rand_seed 7 \
`#--save_model` \
--models_dir "models/" \
--train_file "data/english/train/wsj_train.conll06" \
--train_slice 100 \
--print_every 50 \