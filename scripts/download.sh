mkdir benchmark/data
mkdir benchmark/data/infinite-bench
mkdir benchmark/data/longbench

python benchmark/download.py

cd benchmark/data/infinite-bench
wget https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/math_find.jsonl
wget https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/number_string.jsonl
wget https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/passkey.jsonl
wget https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/longbook_choice_eng.jsonl
wget https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/kv_retrieval.jsonl
wget https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/code_debug.jsonl