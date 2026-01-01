import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.signal import savgol_filter

def parse_log(filename):
    """
    Parses the log file to extract Epoch and Accuracy.
    Robust to irregular line breaks.
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find all pairs of (integer, float)
    # This handles cases like "24 \n 0.53125" or "25 0.5625"
    matches = re.findall(r'(\d+)\s+(\d+\.\d+)', content)
    
    epochs = []
    accuracies = []
    
    for epoch, acc in matches:
        epochs.append(int(epoch))
        accuracies.append(float(acc))
        
    # Sort by epoch just in case
    data = sorted(zip(epochs, accuracies))
    if not data:
        return np.array([]), np.array([])
        
    epochs, accuracies = zip(*data)
    return np.array(epochs), np.array(accuracies)

# Load the data
file_baseline = 'log_subject5_baseline.txt'
file_mish = 'log_subject5_mish.txt'

epochs_base, acc_base = parse_log(file_baseline)
epochs_mish, acc_mish = parse_log(file_mish)

# Check if data loaded correctly
if len(epochs_base) == 0 or len(epochs_mish) == 0:
    print("Error: Could not extract data from logs.")
else:
    # Plotting
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Smoothing function
    def smooth(y, window_size=21, poly_order=3):
        if len(y) < window_size:
            window_size = len(y) if len(y) % 2 == 1 else len(y) - 1
        if window_size < 5: return y
        return savgol_filter(y, window_size, poly_order)

    # Plot Baseline
    ax.plot(epochs_base, acc_base, color='gray', alpha=0.2, label='_nolegend_') # Raw noise
    ax.plot(epochs_base, smooth(acc_base), color='#1f77b4', linewidth=2, label='Baseline (ELU)')

    # Plot Mish
    ax.plot(epochs_mish, acc_mish, color='lightcoral', alpha=0.2, label='_nolegend_') # Raw noise
    ax.plot(epochs_mish, smooth(acc_mish), color='#d62728', linewidth=2, label='Ours (Mish)')

    # Labels and Title
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training Convergence Comparison (Subject 5)', fontsize=14, pad=15)
    
    # Grid and Legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12, loc='lower right', frameon=True)

    # Annotate the gap (max accuracy)
    max_base = np.max(acc_base)
    max_mish = np.max(acc_mish)
    
    # Add text box with statistics
    textstr = '\n'.join((
        r'$\mathrm{Best\ Accuracy:}$',
        r'Baseline: %.2f%%' % (max_base * 100, ),
        r'Mish: %.2f%%' % (max_mish * 100, ),
        r'$\Delta$: +%.2f%%' % ((max_mish - max_base) * 100, )))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Save
    plt.tight_layout()
    plt.savefig('s5_convergence_comparison.png', dpi=300)