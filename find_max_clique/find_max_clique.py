"""Run fmc executable and print max clique nodes
"""

import os

output = os.popen("fmc/src/fmc fmc/testgraphs/hamming6-2.clq.mtx -t 1 -p").read()

nodes = []
for line in output.splitlines():
    token = "Maximum clique:"
    if line.startswith(token):
        nodes = [int(x) for x in line[len(token):].split()]
nodes.sort()
print(nodes)