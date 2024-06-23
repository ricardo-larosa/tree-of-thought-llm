import argparse
from tot.methods.bfs import solve
from tot.tasks.swe import BigCodeTask
from datasets import load_dataset
import json
import time

print("Downloading dataset...")
dataset = load_dataset("bigcode/bigcodebench", split = "v0.1.0_hf", cache_dir='datasets_cache')
preds_path = "BigCode-llama3-70b-8192.jsonl"
try:
    with open(preds_path, "r") as file:
        preds_jsonl = file.read()
except FileNotFoundError:
    preds_jsonl = ""

def update_jsonl(instance_id, model_patch, jsonl_object):
    data = [json.loads(line) for line in jsonl_object.splitlines()]

    new_obj = {
        "task_id": instance_id,
        "solution": model_patch,
    }

    data.append(new_obj)
    updated_jsonl = "\n".join([json.dumps(obj) for obj in data])

    return updated_jsonl

def save_jsonl(jsonl_object, file_path):
    with open(file_path, "w") as file:
        file.write(jsonl_object) if jsonl_object.strip() else file.write("")

args = argparse.Namespace(
    backend='llama3-70b-8192',
    temperature=0.7, 
    task='bigcode', 
    naive_run=False,
    prompt_sample='standard', 
    method_generate='sample', 
    method_evaluate='vote', 
    method_select='greedy', 
    n_generate_sample=5, 
    n_evaluate_sample=5, 
    n_select_sample=1)

print("Solving...")
task = BigCodeTask(dataset)


for index in range(3,4):
    task_id = dataset[index]["task_id"]
    size = len(dataset[index]["instruct_prompt"])
    print(f" ### Task {index} -- {task_id} -> size ({size} )###")
    ys, infos, _ = solve(args, task, index, to_print=False)
    preds_jsonl = update_jsonl(dataset[index]["task_id"], BigCodeTask.parse_diff_block(ys[0]), args.backend, preds_jsonl)
    save_jsonl(preds_jsonl, preds_path)
    print("-----------Generated----------------------")
    print(BigCodeTask.parse_diff_block(ys[0]))
    print("-----------Canonical_Solution-------------")
    print(dataset[index]["canonical_solution"])
    # time.sleep(1)

