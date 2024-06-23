from typing import TypedDict
from typing import cast
from datasets import load_dataset, Dataset

class BigCodeBenchInstance(TypedDict):
    task_id: str
    complete_prompt: str
    instruct_prompt: str
    canonical_solution: str
    code_prompt: str
    test: str
    entry_point: str
    doc_struct: str
    libs: str

def get_dataset() -> list[BigCodeBenchInstance]:
    dataset = cast(Dataset, load_dataset("bigcode/bigcodebench", split="v0.1.0_hf"))
    return [cast(BigCodeBenchInstance, instance) for instance in dataset]

