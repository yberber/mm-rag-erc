

import json

file_path = "PROCESSED_DATASET/MELD/k12_n1/test.json"

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

len(data)

print(data[0]["input"])

inputs = [d["input"] for d in data]

lenghts = [len(i) for i in inputs]
print(lenghts)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

counts, bins = np.histogram(lenghts)
plt.stairs(counts, bins)
plt.show()

plt.hist(lenghts)
plt.show()
