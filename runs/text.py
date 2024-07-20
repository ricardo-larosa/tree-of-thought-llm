import argparse
from tot.methods.bfs import solve
from tot.tasks.text import TextTask

args = argparse.Namespace(backend='gpt-4', 
                          temperature=1.0, 
                          task='text', 
                          naive_run=False, 
                          prompt_sample='cot', 
                          method_generate='sample', 
                          method_evaluate='vote', 
                          method_select='greedy',
                          n_generate_sample=1, 
                          n_evaluate_sample=3, 
                          n_select_sample=5)

task = TextTask()
ys, infos = solve(args,task,0)
print(ys[0])