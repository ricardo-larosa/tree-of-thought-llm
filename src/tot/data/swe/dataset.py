from typing import TypedDict
from typing import cast
from datasets import load_dataset, Dataset

class SwebenchInstance(TypedDict):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str

def get_dataset() -> list[SwebenchInstance]:
    dataset = cast(Dataset, load_dataset("princeton-nlp/SWE-bench", split="dev+test"))
    return [cast(SwebenchInstance, instance) for instance in dataset]

def get_categories() -> list[str]:
    dataset = cast(Dataset, load_dataset("princeton-nlp/SWE-bench", split="dev+test"))
    return list(set([instance["category"] for instance in dataset]))

def main():
    swe_dataset = load_dataset("princeton-nlp/SWE-bench", split="dev+test")
    # print each key-value pair in the first instance
    for key, value in swe_dataset[0].items():
        print(f"{key}: {value}")
    # print a unique list of values for the key "repo"
    print(set([instance["repo"] for instance in swe_dataset]))
    # instances of the dataset which key "repo" is "pytest-dev/pytest"
    pytest_repo = [instance for instance in swe_dataset if instance["repo"] == "pytest-dev/pytest"]
