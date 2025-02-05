# Log File Processing and Spreadsheet Generation

This folder contains scripts to help you combine multiple log files, process them into a structured spreadsheet, and save the results as a CSV file. Follow the detailed steps below to achieve this using the provided scripts.

## Overview

You will use the `combine_logs.py` script to concatenate multiple log files into a single `.txt` file, then process that file into a structured spreadsheet. The spreadsheet will include operations like inserting rows, creating a pivot table, and performing data aggregation. Finally, the data will be saved as a CSV file named `input_data.csv`.

## Steps to Process Log Files and Create Spreadsheet

### Step 1: Combine Log Files into a Single TXT
1. Use the `combine_logs.py` script to concatenate all log files into a single `.txt` file. This script will combine all your logs with a space-separated delimiter.

   Run the script as follows:

   ```bash
   python combine_logs.py
   ```
   This will produce a combined_logs.txt file containing the concatenated data from your logs.

### Step 2: Create a Spreadsheet from the TXT File

    Open the combined_logs.txt file in a spreadsheet tool like Microsoft Excel or Google Sheets.
    Use the space-separated delimiter to split the data into columns.

### Step 3: Insert Row Above and Add Identifiers

    Insert a row at the very top of the sheet.
    Add appropriate identifiers to the columns, such as "Matrix", "Preset", and "Cut". These identifiers will be helpful for organizing the data later.

### Step 4: Create a Pivot Table

    Select the range of data that you want to include in the pivot table.
    Create a pivot table with the following configurations:
        Row Fields: "Matrix" and "Preset" labels (ensure they are repeated).
        Data Field: "Cut", using the Average aggregate function.

### Step 5: Adjust Cut Column

    If necessary, eliminate the decimal places in the "Cut" average column. Convert the values to whole integers.

### Step 6: Copy and Paste Pivot Table as Raw Text

    If necessary, copy all elements of the pivot table.
    Paste them into a new spreadsheet as raw, unformatted text. This will help in maintaining the structure and formatting without relying on any pivot table settings.

### Step 7: Save the Final Sheet as CSV

    After processing, save the final sheet as a CSV file.
    In LibreOffice, go to File > Save As and select the .csv format. Name the file input_data.csv.

### Step 8: Use Additional Scripts for Filtering

    If you need to filter data before combining logs or creating the spreadsheet, you can use other available scripts to filter data according to your specific needs before proceeding with the steps above.
