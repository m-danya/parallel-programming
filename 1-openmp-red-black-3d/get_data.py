# скрипт для вывода списка в синтаксисе питона из полученных чисел, разделённых \n

import pprint
import sys

list = []

for line in sys.stdin:
    try:
        list.append(float(line.strip()))
    except:
        break

pprint.pprint(list, compact=True)