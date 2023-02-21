# Make prediction
# Write prediction to file
# Evaluate prediction against gold labels
# Print UAS
# comment lines within command with #`bool_arg` to avoid breaking

python3 src/main.py \
--language "english" \
--mode "dev" \
--model_file "models/english-3-0.3-zeros_14021717.pkl" \
--models_dir "models/" \
--test_file "data/english/dev/wsj_dev.conll06.blind" \
--test_slice 0 \
--print_every 200 \
--preds_dir "preds/" \
`#--save_preds` \

python3 src/evaluation.py \
--mode "dev" \
--language "english" \
--pred_file "preds/dev_english-3-0.3-zeros_14021717.pkl.conll06.pred" \
--gold_file "data/english/dev/wsj_dev.conll06.gold" \
--metrics "uas,ucm" \