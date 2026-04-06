import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os, sys

pfx_cost_200k = 0.0515

compaction_costs_200k = [0.0265, 0.0515, 0.0765, 0.1015, 0.1265]
compaction_lengths_200k = [10000, 20000, 30000, 40000, 50000]
compaction_lengths_200_percentages = [length / 200000 * 100 for length in compaction_lengths_200k]

# Plot pfx cost as a horizontal line
plt.axhline(y=pfx_cost_200k, color='r', linestyle='--', label='Prefix Cache Cost')
# Plot compaction costs
plt.plot(compaction_lengths_200_percentages, compaction_costs_200k, marker='o', label='Compaction Cost')
plt.xlabel('Compaction Length (%)')
plt.ylabel('$ Cost')
plt.title('Prefix Cache vs Compaction on 200k input length\n(GPT 5.4)')
plt.legend()

save_folder = '../figures'
today = datetime.now().strftime('%Y-%m-%d')
save_path = os.path.join(save_folder, today)
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, 'prefix_cache_vs_compaction_200k_cost.pdf'), format='pdf', bbox_inches='tight')

pfx_cost_500k = 0.25225

compaction_costs_500k = [0.0265, 0.0515, 0.0765, 0.1015, 0.1265, 0.1515,
    0.1765,   
    0.2015,   
    0.2265,   
    0.2515,  
    0.3765,  
    0.5015,
]

compaction_lengths_500k = [10000, 20000, 30000, 40000, 50000, 60000,
    70000,
    80000,
    90000,
    100000,
    150000,
    200000,
]

compaction_lengths_500k_percentages = [length / 500000 * 100 for length in compaction_lengths_500k]

# Plot pfx cost as a horizontal line
plt.figure()
plt.axhline(y=pfx_cost_500k, color='r', linestyle='--', label='Prefix Cache Cost')
# Plot compaction costs
plt.plot(compaction_lengths_500k_percentages, compaction_costs_500k, marker='o', label='Compaction Cost')
plt.xlabel('Compaction Length (%)')
plt.ylabel('$ Cost')
plt.title('Prefix Cache vs Compaction on 500k input length\n(GPT 5.4)')
plt.legend()
plt.savefig(os.path.join(save_path, 'prefix_cache_vs_compaction_500k_cost.pdf'), format='pdf', bbox_inches='tight')



