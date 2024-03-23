import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=None)
    args = parser.parse_args()
    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    datasets_list = [s.strip() for s in datasets_list]
    args.datasets = datasets_list
    return args

if __name__ == "__main__":
    args = parse_args()
    for dataset in args.datasets:

        out_path = os.path.join(
            args.output_dir_path,
            f"{dataset}.jsonl"
        )

        lines = []
        for rank in range(args.world_size):
            file_path = out_path + f"_{rank}"
            f = open(file_path, "r")
            lines += f.readlines()
            f.close()

        lines = [l.strip() for l in lines]
        f = open(out_path, "w+")
        f.write(
            "\n".join(lines)
        )
        f.close()

        for rank in range(args.world_size):
            file_path = out_path + f"_{rank}"
            os.remove(file_path)