import pandas as pd
import sys
import numpy as np

def extract_weight_changes(input_file, output_file):
    """
    Extract only the weight changes from a synapse recording file and calculate
    rates of weight updates (changes per simulation time) for each synapse.
    
    Args:
        input_file: Path to the tab-delimited input file
        output_file: Path where to save the output file
    """
    # Read the file with tab delimiter and no header
    df = pd.read_csv(input_file, delimiter='\t', header=None)
    
    # Keep only the first 4 columns (drop the empty NaN column if it exists)
    if len(df.columns) > 4:
        df = df.iloc[:, :4]
    
    # Rename columns
    df.columns = ['id_granule', 'id_pc', 'time', 'weight']
    
    SIMULATION_TIME = 15000  # Fixed simulation time as specified
    
    # Get simulation time range (for information only)
    first_spike_time = df['time'].min()
    last_spike_time = df['time'].max()
    total_simulation_time = last_spike_time - first_spike_time
    
    # Create a unique identifier for each synapse
    df['synapse_id'] = df['id_granule'].astype(str) + '_' + df['id_pc'].astype(str)
    
    # Sort by synapse_id and time
    df = df.sort_values(['synapse_id', 'time'])
    
    # Track weight changes
    last_weights = {}
    changed_indices = []
    synapse_update_counts = {}  # Count of weight updates per synapse
    total_synapses = set()  # Keep track of all unique synapses
    
    for idx, row in df.iterrows():
        synapse = row['synapse_id']
        current_weight = row['weight']
        
        # Add to total synapses set
        total_synapses.add(synapse)
        
        # Check if this is a new synapse or if the weight has changed
        if synapse not in last_weights:
            # First appearance of this synapse
            last_weights[synapse] = current_weight
        elif current_weight != last_weights[synapse]:
            # Weight has changed - add to our results
            changed_indices.append(idx)
            
            # Increment update count for this synapse
            if synapse not in synapse_update_counts:
                synapse_update_counts[synapse] = 1
            else:
                synapse_update_counts[synapse] += 1
                
            # Update last weight
            last_weights[synapse] = current_weight
    
    # Get only the rows where weight changed
    weight_changes_df = df.loc[changed_indices].copy() if changed_indices else pd.DataFrame(columns=df.columns)
    
    # Calculate rate for each synapse (updates/simulation_time)
    synapse_rates = {}
    total_rates_synapse = []
    for synapse in total_synapses:
        # If this synapse had updates, calculate its rate
        if synapse in synapse_update_counts:
            rate = synapse_update_counts[synapse] / SIMULATION_TIME
        else:
            # No updates for this synapse
            rate = 0
        synapse_rates[synapse] = rate
        total_rates_synapse.append(rate)
    
    #total_rates_synapse = np.nonzero(total_rates_synapse)
    median_weight_rate = np.median(total_rates_synapse)
    q1_weight_rate = np.percentile(total_rates_synapse, 25, axis=0)
    q3_weight_rate = np.percentile(total_rates_synapse, 75, axis=0)

    std_weight_rate = np.std(total_rates_synapse)

    # Calculate average rate across all synapses
    total_updates = sum(synapse_update_counts.values())
    avg_rate_per_synapse = total_updates / (len(total_synapses) * SIMULATION_TIME) if total_synapses else 0
    
    # Alternative calculation: total updates divided by simulation time
    total_update_rate = total_updates / SIMULATION_TIME if SIMULATION_TIME > 0 else 0
    
    # Save to output file (either CSV or TXT)
    if output_file.endswith('.csv'):
        weight_changes_df.to_csv(output_file, index=False)
    else:
        # For TXT file, include simulation time info at the top
        with open(output_file, 'w') as f:
            f.write(f"Simulation time analysis:\n")
            f.write(f"First spike time: {first_spike_time}\n")
            f.write(f"Last spike time: {last_spike_time}\n")
            f.write(f"Total simulation time: {total_simulation_time}\n")
            f.write(f"Fixed simulation time used for rate calculations: {SIMULATION_TIME}\n")
            f.write(f"Total weight changes: {total_updates}\n")
            f.write(f"Total unique synapses: {len(total_synapses)}\n")
            f.write(f"Weight changes per simulation time unit: {total_update_rate:.6f}\n\n")
            
            # Write rate information
            f.write("Weight change rate analysis:\n")
            f.write(f"Average rate of weight updates per synapse: {avg_rate_per_synapse:.6f}\n\n")
            f.write(f"Median weight update rate: {median_weight_rate:.6f}\n")
            f.write(f"First quartile of weight update rate: {q1_weight_rate:.6f}\n")
            f.write(f"Third quartile of weight update rate: {q3_weight_rate:.6f}\n")
            
            f.write("Rate of weight updates by synapse (changes/simulation_time):\n")
            for synapse, rate in sorted(synapse_rates.items()):
                updates = synapse_update_counts.get(synapse, 0)
                f.write(f"{synapse}: {updates} updates, rate = {rate:.6f}\n")
            f.write("\n")
            
            f.write("Weight changes:\n")
            if not weight_changes_df.empty:
                weight_changes_df.to_string(f, index=False)
            else:
                f.write("No weight changes detected.\n")
    
    # Print summary info
    print(f"Analysis complete:")
    print(f"- First spike time: {first_spike_time}")
    print(f"- Last spike time: {last_spike_time}")
    print(f"- Total simulation time: {total_simulation_time}")
    print(f"- Fixed simulation time used for rates: {SIMULATION_TIME}")
    print(f"- Total weight changes detected: {total_updates}")
    print(f"- Total unique synapses: {len(total_synapses)}")
    print(f"- Average rate of weight updates per synapse: {avg_rate_per_synapse:.6f}")
    print(f"- Total weight changes per simulation time: {total_update_rate:.6f}")
    print(f"Median weight update rate: {median_weight_rate:.6f}")
    print(f"First quartile of weight update rate: {q1_weight_rate:.6f}\n")
    print(f"Third quartile of weight update rate: {q3_weight_rate:.6f}\n")
    print(f"- Results saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_weight_changes.py input_file.csv output_file.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    extract_weight_changes(input_file, output_file)