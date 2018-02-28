
if [ "$1" == "data" ]; then
	python preprocess.py \
		-train_src ./weibo/src-train.keywords.txt \
		-train_tgt ./weibo/tgt-train.keywords.txt \
		-valid_src ./weibo/src-valid.keywords.txt \
		-valid_tgt ./weibo/tgt-valid.keywords.txt \
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
		-learning_rate 0.001 \
		-gpuid 0 \
		-exp_host 172.18.217.118 \
		-exp key
elif [ "$1" == "predict" ]; then
	python translate.py \
		-model keyphrase.pt \
		-src ./weibo/src-test.keywords.txt \
		-tgt ./weibo/tgt-test.keywords.txt \
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
