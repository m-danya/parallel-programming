from collections import defaultdict
import re
import os


# scp 'polus:*txt' results; python3 parse_output.py


files = [f for f in os.listdir("results") if f.endswith(".txt") and "err" not in f]

r = defaultdict(list)

for filename in files:
    with open("results/" + filename, "r") as file:
        content = file.readlines()
        # f"{current_time_str}_try_{x}_out_{n}_{processes_number}_{threads_number}.txt"
        *extra, n_value, processes, threads = filename.replace(".txt", "").split("_")
        time_elapsed = None
        for line in content:
            if line.startswith("Running with N"):
                n_value = line.replace("(SEQUENTIAL VERSION)", "").strip().split()[-1]
            if line.startswith("Max eps"):
                max_eps = line.split()[-1]
            if line.startswith("Elapsed time"):
                time_elapsed = line.split()[-2]

        result = f"{max_eps = }, {time_elapsed = }"

        # if threads == "4":
        # if threads == "1" and processes != "1":
        # continue

        r[f"{processes = }, {threads = }, N = {n_value}"].append(result)

from pprint import pprint

pprint(r)
