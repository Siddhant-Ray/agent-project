#!/bin/bash

seeds=100

rm -r ./traces/*.txt

for i in $(seq 1 $seeds); do
  python3 ./generator_v2.py --seed $i
done