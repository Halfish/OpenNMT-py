
if [ "$1" == "data" ]; then
	python preprocess.py \
		-train_src ./weibo/src-train.txt \
		-train_tgt ./weibo/tgt-train.txt \
		-valid_src ./weibo/src-valid.txt \
		-valid_tgt ./weibo/tgt-valid.txt \
		-save_data data/weibo \
		-max_shard_size 64000000 \
		-src_words_min_frequency 10 \
		-tgt_words_min_frequency 10 \
		-src_seq_length 100 \
		-tgt_seq_length 20 \
		-share_vocab
elif [ "$1" == "train" ]; then
	python train.py \
		-data data/weibo \
		-save_model baseline \
		-optim adam \
		-gpuid 0 \
		-learning_rate 0.001 \
		-exp_host 172.18.217.118 \
		-exp base
elif [ "$1" == "predict" ]; then
	python translate.py \
		-model baseline.pt \
		-src ./weibo/src-test.txt \
		-tgt ./weibo/tgt-test.txt \
		-output pred.txt \
		-gpu 0 \
		-replace_unk \
		-report_bleu \
		-report_rouge \
		-verbose \
		-attn_debug
else
	echo "wrong parameter!"
fi
