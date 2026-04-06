import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os, sys

save_folder = '../figures'
today = datetime.now().strftime('%Y-%m-%d')
save_path = os.path.join(save_folder, today)
os.makedirs(save_path, exist_ok=True)

pfx_cost_500k = 61.03515625

compaction_costs_500k = [ 1.220703125, 2.44140625,   
    6.103515625,    
    9.1552734375,  
    12.20703125,
    24.4140625,
    36.62109375,
]

compaction_lengths_500k = [10000, 20000, 50000, 75000, 100000, 200000, 300000]

compaction_lengths_500k_percentages = [length / 500000 * 100 for length in compaction_lengths_500k]

# Plot pfx cost as a horizontal line
plt.figure()
plt.axhline(y=pfx_cost_500k, color='r', linestyle='--', label='Prefix Cache KV Cache Size')
# Plot compaction costs
plt.plot(compaction_lengths_500k_percentages, compaction_costs_500k, marker='o', label='Compaction KV Cache Size')
plt.xlabel('Compaction Length (%)')
plt.ylabel('KV Cache Size (GB)')
plt.title('Prefix Cache vs Compaction on 500k input length\n(LLama-3.1-Nemotron-8B-UltraLong-1M-Instruct)')
plt.legend()
plt.savefig(os.path.join(save_path, 'prefix_cache_vs_compaction_500k_kv.pdf'), format='pdf', bbox_inches='tight')



