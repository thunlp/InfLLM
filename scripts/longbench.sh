config=config/mistral-inf-llm.yaml

datasets="narrativeqa,qasper,multifieldqa_en,\
hotpotqa,2wikimqa,musique,\
gov_report,qmsum,multi_news,\
trec,triviaqa,samsum,\
passage_count,passage_retrieval_en,\
lcc,repobench-p"

mkdir benchmark/longbench-result

python benchmark/pred.py \
--config_path ${config} \
--output_dir_path benchmark/longbench-result \
--datasets ${datasets} 

python benchmark/eval.py --dir_path benchmark/longbench-result