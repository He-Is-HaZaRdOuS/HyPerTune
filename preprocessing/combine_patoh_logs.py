import os
import re

output_dir = "../data/partitions/"
log_dir = "../data/logs/patoh_logs"
output_file = f"{output_dir}combined_patoh_logs.txt"


def get_filename(path):
    return os.path.basename(path)  # Extracts only the filename


def parse_log_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Extract relevant information using regular expressions
    hypergraph = re.search(r"Hypergraph\s*:\s*(\S+)", content)
    num_cells = re.search(r"#Cells\s*:\s*(\d+)", content)
    num_nets = re.search(r"#Nets\s*:\s*(\d+)", content)
    num_pins = re.search(r"#Pins\s*:\s*(\d+)", content)
    partition_count = re.search(r"(\d+)-way partitioning results", content)
    cost = re.search(r"Cost:\s*(\d+)", content)
    min_weight = re.search(r"Min=\s*(\d+)", content)
    max_weight = re.search(r"Max=\s*(\d+)", content)
    partition_time = re.search(r"Total\s*:\s*([\d.]+)\s*sec", content)
    total_time = re.search(r"Total \(w I/O\):\s*([\d.]+)\s*sec", content)
    seed = re.search(r"SD=([-\d]+)", content)
    imbalance = re.search(r"IB=([\d.]+)", content)

    # Extract Command line
    command_line = re.search(r"Command:\s*(.+)", content)
    command_dict = {}

    return {
        "Hypergraph": get_filename(hypergraph.group(1))
        if hypergraph
        else "N/A",
        "#Cells": num_cells.group(1) if num_cells else "N/A",
        "#Nets": num_nets.group(1) if num_nets else "N/A",
        "#Pins": num_pins.group(1) if num_pins else "N/A",
        "Partitions": partition_count.group(1) if partition_count else "N/A",
        "Cost": cost.group(1) if cost else "N/A",
        "Min Weight": min_weight.group(1) if min_weight else "N/A",
        "Max Weight": max_weight.group(1) if max_weight else "N/A",
        "Partition Time (s)": partition_time.group(1)
        if partition_time
        else "N/A",
        "Total Time (s)": total_time.group(1) if total_time else "N/A",
        "Seed": seed.group(1) if seed else "N/A",
        "Imbalance Ratio": imbalance.group(1) if imbalance else "N/A",
        "Command": command_dict,  # Command line parameters
    }


def concatenate_logs(log_dir, output_file):
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    summaries = []

    for log_file in log_files:
        file_path = os.path.join(log_dir, log_file)
        summary = parse_log_file(file_path)
        summaries.append(summary)

    with open(output_file, "w") as outfile:
        for summary in summaries:
            outfile.write(f"Hypergraph: {summary['Hypergraph']} ")
            outfile.write(f"#Cells: {summary['#Cells']} ")
            outfile.write(f"#Nets: {summary['#Nets']} ")
            outfile.write(f"#Pins: {summary['#Pins']} ")
            outfile.write(f"Partitions: {summary['Partitions']} ")
            outfile.write(f"Cost: {summary['Cost']} ")
            outfile.write("Preset: PaToH-v3.3 ")
            outfile.write(f"Min Weight: {summary['Min Weight']} ")
            outfile.write(f"Max Weight: {summary['Max Weight']} ")
            outfile.write(
                f"Partition Time (s): {summary['Partition Time (s)']} "
            )
            outfile.write(f"Total Time (s): {summary['Total Time (s)']} ")
            outfile.write(f"Seed: {summary['Seed']} ")
            outfile.write(f"Imbalance Ratio: {summary['Imbalance Ratio']} ")

            # Print command-line parameters
            outfile.write("Command Parameters: ")
            for key, value in summary["Command"].items():
                outfile.write(f"  {key}: {value} ")

            outfile.write("\n")


def main():
    concatenate_logs(log_dir, output_file)


if __name__ == "__main__":
    main()
#
