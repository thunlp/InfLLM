SHORT=w:,f:,d:,o:,h
LONG=world_size:,config_path:,datasets:,output_dir_path:,help

PARSED=$(getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@")

if [[ $? -ne 0 ]]; then
    echo "Invalid Arguments."
    exit 2
fi

eval set -- "$PARSED"

world_size=1
datasets=passkey,number_string,code_debug,math_find,longbook_choice_eng
output_dir_path=benchmark/benchmark-result

while true; do
    case "$1" in
        -h|--help)
            echo "Usage: $0 [--world_size <int>] [--config_path <file>] [--datasets <dataset_name>] [--output_dir_path <dir>] [--help]"
            exit
            ;;
        -w|--world_size)
            world_size="$2"
            shift 2
            ;;
        -f|--config_path)
            config_path="$2"
            shift 2
            ;;
        -d|--datasets)
            datasets="$2"
            shift 2
            ;;
        -o|--output_dir_path)
            output_dir_path="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done


mkdir -p ${output_dir_path}

trap 'kill $(jobs -p)' SIGINT

for ((rank=0; rank < $world_size; ++rank))
do
    CUDA_VISIBLE_DEVICES=${rank} python benchmark/pred.py \
    --config_path ${config_path} \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} \
    --world_size ${world_size} \
    --rank ${rank} &
    echo "worker $rank started"
done

wait

python benchmark/merge.py \
    --output_dir_path ${output_dir_path} \
    --datasets ${datasets} \
    --world_size ${world_size}

python benchmark/eval.py --dir_path ${output_dir_path}