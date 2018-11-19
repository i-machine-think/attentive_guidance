#! /bin/sh

TRAIN_PATH=test/test_data/train_small.txt
DEV_PATH=test/test_data/dev_small.txt
LOOKUP=test/test_data/lookup_small.txt
LOOKUP_HARD_ATTN_WITH_EOS=test/test_data/lookup_small_attn_with_eos.txt
LOOKUP_HARD_ATTN_WITHOUT_EOS=test/test_data/lookup_small_attn_without_eos.txt
EXPT_DIR=test_exp

mkdir $EXPT_DIR

# use small parameters for quicker testing
EMB_SIZE=2
H_SIZE=4
CELL='lstm'
CELL2='gru'
EPOCH=2
CP_EVERY=3

EX=0
ERR=0

echo "\n\nTest training with hard attention without EOS"
python3 train_model.py --train $LOOKUP_HARD_ATTN_WITHOUT_EOS --dev $LOOKUP_HARD_ATTN_WITHOUT_EOS --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL2 --attention 'post-rnn' --epoch $EPOCH --save_every $CP_EVERY --attention_method 'hard'  --ignore_output_eos
ERR=$((ERR+$?)); EX=$((EX+1))

echo "\n\nTest training with pre-rnn learned hard attention with EOS"
python3 train_model.py --train $LOOKUP_HARD_ATTN_WITH_EOS --dev $LOOKUP_HARD_ATTN_WITH_EOS --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL2 --attention 'pre-rnn' --epoch $EPOCH --save_every $CP_EVERY --attention_method 'mlp' --use_attention_loss --scale_attention_loss 1
ERR=$((ERR+$?)); EX=$((EX+1))

echo "\n\nTest training with post-rnn learned hard attention with EOS"
python3 train_model.py --train $LOOKUP_HARD_ATTN_WITH_EOS --dev $LOOKUP_HARD_ATTN_WITH_EOS --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL2 --attention 'post-rnn' --epoch $EPOCH --save_every $CP_EVERY --attention_method 'mlp' --use_attention_loss --scale_attention_loss 1
ERR=$((ERR+$?)); EX=$((EX+1))

echo "\n\n\n$EX tests executed, $ERR tests failed\n\n"

rm -r $EXPT_DIR
