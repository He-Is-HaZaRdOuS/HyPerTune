import os
import re

log_dir = '/home/hazardous/Desktop/HyPerTune-tools/kahypar_logs'
output_file = 'combined_kahypar_logs.txt'

def get_filename(path):
    return os.path.basename(path)  # Extracts only the filename

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract relevant information using regular expressions
    hypergraph = re.search(r'Hypergraph:\s+(.+)', content)
    k = re.search(r'k:\s+(\d+)', content)
    epsilon = re.search(r'epsilon:\s+([\d.]+)', content)
    seed = re.search(r'seed:\s+(-?\d+)', content)  # Supports negative values
    partition_time = re.search(r'Partition time\s+=\s+([\d.]+)\s+s', content)
    hyperedge_cut = re.search(r'Hyperedge Cut\s+\(minimize\)\s+=\s+(\d+)', content)
    imbalance = re.search(r'Imbalance\s+=\s+([\d.]+)', content)

    # Extract partition sizes
    partition_sizes = re.findall(r'\|part\s+\d+\s+\|\s+=\s+(\d+)\s+w\(\s+\d+\s+\)\s+=\s+\d+', content)

    # Extract Command line
    command_line = re.search(r'Command:\s+(.+)', content)
    command_dict = {}
    preset = 'N/A'
    if command_line:
        command_parts = command_line.group(1).split()
        for i in range(len(command_parts)):
            if command_parts[i] == '-p' and i + 1 < len(command_parts):
                preset = get_filename(command_parts[i + 1])  # Extract preset filename
            if i % 2 == 0 and i + 1 < len(command_parts):
                command_dict[command_parts[i]] = get_filename(command_parts[i + 1])

    return {
        'Hypergraph': get_filename(hypergraph.group(1)) if hypergraph else 'N/A',
        'Preset': preset,
        'k': k.group(1) if k else 'N/A',
        'epsilon': epsilon.group(1) if epsilon else 'N/A',
        'seed': seed.group(1) if seed else 'N/A',
        'Partition Time (s)': partition_time.group(1) if partition_time else 'N/A',
        'Hyperedge Cut': hyperedge_cut.group(1) if hyperedge_cut else 'N/A',
        'Imbalance': imbalance.group(1) if imbalance else 'N/A',
        'Partition Sizes': ', '.join(partition_sizes) if partition_sizes else 'N/A',
        'Command': command_dict  # Command line parameters
    }

def concatenate_logs(log_dir, output_file):
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    summaries = []

    for log_file in log_files:
        file_path = os.path.join(log_dir, log_file)
        summary = parse_log_file(file_path)
        summaries.append(summary)

    with open(output_file, 'w') as outfile:
        for summary in summaries:
            outfile.write(f"Hypergraph: {summary['Hypergraph']} ")
            outfile.write(f"Preset: {summary['Preset']} ")
            outfile.write(f"k: {summary['k']} ")
            outfile.write(f"Epsilon: {summary['epsilon']} ")
            outfile.write(f"Seed: {summary['seed']} ")
            outfile.write(f"Partition Time (s): {summary['Partition Time (s)']} ")
            outfile.write(f"Hyperedge Cut: {summary['Hyperedge Cut']} ")
            outfile.write(f"Imbalance: {summary['Imbalance']} ")
            outfile.write(f"Partition Sizes: {summary['Partition Sizes']} ")

            # Print command-line parameters
            outfile.write("Command Parameters: ")
            for key, value in summary['Command'].items():
                outfile.write(f"  {key}: {value} ")

            outfile.write("\n")

def main():
    concatenate_logs(log_dir, output_file)

if __name__ == "__main__":
    main()
