standard_prompt = '''
{input}
'''

cot_prompt = '''
{input}

Important Instruction: Your output must be of the following format:

Plan:

Your plan here.

Code Block:

```python
Your code here.
```

Do not provide any explanation or context in the code block. Only the code should be present.
'''


vote_prompt = '''Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where {s} the integer id of the choice.
'''

compare_prompt = '''Briefly analyze the degree of correctness of the following two code blocks. Conclude in the last line "The more correct code blocks is 1", "The more correct code block is 2", or "The two code blocks are equally correct".
'''

score_prompt = '''Analyze the following code block, then at the last line conclude "Therefore the correctness score is {s}", where {s} is an integer from 1 to 10.
'''