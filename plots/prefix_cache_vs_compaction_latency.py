import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os, sys

save_folder = '../figures'
today = datetime.now().strftime('%Y-%m-%d')
save_path = os.path.join(save_folder, today)
os.makedirs(save_path, exist_ok=True)

pfx_cost_500k = 0.76

compaction_costs_500k = [0.13, 0.26, 0.759, 1.35, 2.024, 6.20]

compaction_lengths_500k = [10000, 20000, 50000, 75000, 100000, 200000]

compaction_lengths_500k_percentages = [length / 500000 * 100 for length in compaction_lengths_500k]

# Plot pfx cost as a horizontal line
plt.figure()
plt.axhline(y=pfx_cost_500k, color='r', linestyle='--', label='Prefix Cache Latency')
# Plot compaction costs
plt.plot(compaction_lengths_500k_percentages, compaction_costs_500k, marker='o', label='Compaction Latency')
plt.xlabel('Compaction Length (%)')
plt.ylabel('Prefill Latency (s)')
plt.title('Prefix Cache vs Compaction on 500k input length\n(LLama-3.1-Nemotron-8B-UltraLong-1M-Instruct)')
plt.legend()
plt.savefig(os.path.join(save_path, 'prefix_cache_vs_compaction_500k_latency.pdf'), format='pdf', bbox_inches='tight')



