import pandas as pd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import math
import sys
import json
import os
import itertools
import matplotlib.ticker as ticker
import numpy as np
 
def remove_short_values(d):
    """
    Removes dictionary elements with values of length less than 2.
 
    Args:
        d (dict): Input dictionary.
 
    Returns:
        dict: Dictionary with short values removed.
    """
    return {key: value for key, value in d.items() if len(value) >= 3}
 
def get_unique_values(csv_file):
    df = pd.read_csv(csv_file)
    unique_values = {}
    for column in df.columns:
        values = df[column].unique().tolist()
        values.insert(0, "N/A")  # Add "N/A" option
        unique_values[column] = values
    return remove_short_values(unique_values)
 
 
 
def create_gui(unique_values, df):
    root = tk.Tk()
    root.title("CSV Column Selector")
 
    dropdown_vars = []
    dropdown_widgets = {}  # Dictionary to store dropdown widgets
    max_rows = int((16.0/9.0)*math.sqrt(len(unique_values)))
    num_columns = math.ceil(len(unique_values) / max_rows)
 
    for i, (column, values) in enumerate(unique_values.items()):
        col = i // max_rows
        row = i % max_rows
 
        label = ttk.Label(root, text=column)
        label.grid(row=row, column=col*2, padx=5, pady=5, sticky="w")
 
        var = tk.StringVar(root)
        var.set("N/A")  # Set default value to "N/A"
        dropdown = ttk.Combobox(root, textvariable=var, values=values, state="readonly")
        dropdown.grid(row=row, column=col*2+1, padx=5, pady=5)
        dropdown_vars.append((column, var))
        dropdown_widgets[column] = dropdown  # Store the dropdown widget
 
    # Frame for sorting, X-axis, Y-axis, and plot type selection
    selection_frame = ttk.Frame(root)
    selection_frame.grid(row=max_rows, column=0, columnspan=num_columns*2, pady=10)
 
    # Primary Sort by selection
    sort_label = ttk.Label(selection_frame, text="Primary Sort")
    sort_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
    sort_var = tk.StringVar(root)
    sort_options = ["None"] + list(unique_values.keys())
    sort_dropdown = ttk.Combobox(selection_frame, textvariable=sort_var, values=sort_options, state="readonly")
    sort_dropdown.set("None")  # Set default to "None"
    sort_dropdown.grid(row=0, column=1, padx=5, pady=5)
 
    # Secondary Sort by selection
    secondary_sort_label = ttk.Label(selection_frame, text="Secondary Sort")
    secondary_sort_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
    secondary_sort_var = tk.StringVar(root)
    secondary_sort_dropdown = ttk.Combobox(selection_frame, textvariable=secondary_sort_var, values=sort_options, state="readonly")
    secondary_sort_dropdown.set("None")  # Set default to "None"
    secondary_sort_dropdown.grid(row=0, column=3, padx=5, pady=5)
 
    # Sort order selection
    order_label = ttk.Label(selection_frame, text="Sort order")
    order_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")
    order_var = tk.StringVar(root)
    order_options = ["Ascending", "Descending"]
    order_dropdown = ttk.Combobox(selection_frame, textvariable=order_var, values=order_options, state="readonly")
    order_dropdown.set("Ascending")  # Set default to "Ascending"
    order_dropdown.grid(row=0, column=5, padx=5, pady=5)
 
    # X-axis selection
    x_label = ttk.Label(selection_frame, text="X-axis")
    x_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    x_var = tk.StringVar(root)
    x_options = ["Index","GEMM_SIZE"] + list(unique_values.keys())
    x_dropdown = ttk.Combobox(selection_frame, textvariable=x_var, values=x_options, state="readonly")
    x_dropdown.set("Index")  # Set default to "Index"
    x_dropdown.grid(row=1, column=1, padx=5, pady=5)
 
    # Y-axis selection
    y_label = ttk.Label(selection_frame, text="Y-axis")
    y_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")
    y_var = tk.StringVar(root)
    y_dropdown = ttk.Combobox(selection_frame, textvariable=y_var, values=list(unique_values.keys()), state="readonly")
    y_dropdown.grid(row=1, column=3, padx=5, pady=5)
 
    # Plot type selection
    plot_label = ttk.Label(selection_frame, text="Plot Type")
    plot_label.grid(row=1, column=4, padx=5, pady=5, sticky="w")
    plot_var = tk.StringVar(root)
    plot_options = [
        "Scatter", "Line", "Bar", "Dash", "Step", "Stem", "Area",
        "Filled Line", "Horizontal Bar", "Box Plot", "Violin Plot"
    ]
    plot_dropdown = ttk.Combobox(selection_frame, textvariable=plot_var, values=plot_options, state="readonly")
    plot_dropdown.set("Scatter")  # Set default to "Scatter"
    plot_dropdown.grid(row=1, column=5, padx=5, pady=5)
 
    # Create a list to store plot data
    plots = []
   
    # Create a color cycle for different plots
    color_cycle = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
 
    def on_submit(add_plot=False):
        selected_values = {column: var.get() for column, var in dropdown_vars}
        x_axis = x_var.get()
        y_axis = y_var.get()
        plot_type = plot_var.get()
        primary_sort = sort_var.get()
        secondary_sort = secondary_sort_var.get()
        sort_order = order_var.get()
       
        # print("Selected values:", selected_values)
        # print("X-axis:", x_axis)
        # print("Y-axis:", y_axis)
        # print("Plot type:", plot_type)
        # print("Primary Sort:", primary_sort)
        # print("Secondary Sort:", secondary_sort)
        # print("Sort order:", sort_order)
        plt.ion()
        if not add_plot:
            plt.figure(figsize=(12, 6))
            plots.clear()
       
        color = next(color_cycle)
        plot_data = filter_and_plot(df, selected_values, x_axis, y_axis, plot_type,
                                    primary_sort, secondary_sort, sort_order, color)
        plots.append(plot_data)
       
        # Update the legend
        handles = [plot['handle'] for plot in plots]
        labels = [plot['label'] for plot in plots]
        plt.legend(handles, labels, title="Selected Parameters", loc="best", bbox_to_anchor=(1.05, 1), fontsize='small')
       
        plt.tight_layout()  # Adjust layout to prevent legend from being cut off
        plt.draw()  # Redraw the plot
 
    submit_button = ttk.Button(root, text="New Plot", command=lambda: on_submit(False))
    submit_button.grid(row=max_rows+1, column=0, columnspan=num_columns, pady=10)
 
    add_plot_button = ttk.Button(root, text="Add Plot", command=lambda: on_submit(True))
    add_plot_button.grid(row=max_rows+1, column=num_columns, columnspan=num_columns, pady=10)
 
    def save_selections():
        selections = {
            'dropdown_values': {column: var.get() for column, var in dropdown_vars},
            'x_axis': x_var.get(),
            'y_axis': y_var.get(),
            'plot_type': plot_var.get(),
            'primary_sort': sort_var.get(),
            'secondary_sort': secondary_sort_var.get(),
            'sort_order': order_var.get()
        }
        with open('selections.json', 'w') as f:
            json.dump(selections, f)
        print("Selections saved")
 
    # save_button = ttk.Button(root, text="Save Selections", command=save_selections)
    # save_button.grid(row=max_rows+2, column=0, columnspan=num_columns*2, pady=10)
 
    def restore_selections():
        if os.path.exists('selections.json'):
            try:
                with open('selections.json', 'r') as f:
                    selections = json.load(f)
               
                # Check if saved selections are compatible with current GUI
                if set(selections['dropdown_values'].keys()) != set(dict(dropdown_vars).keys()):
                    raise ValueError("Incompatible column structure in saved selections")
               
                for column, var in dropdown_vars:
                    if column in selections['dropdown_values']:
                        if selections['dropdown_values'][column] in dropdown_widgets[column]['values']:
                            var.set(selections['dropdown_values'][column])
                        else:
                            raise ValueError(f"Invalid value for {column} in saved selections")
               
                if selections['x_axis'] in x_dropdown['values']:
                    x_var.set(selections['x_axis'])
                else:
                    raise ValueError("Invalid X-axis in saved selections")
               
                if selections['y_axis'] in y_dropdown['values']:
                    y_var.set(selections['y_axis'])
                else:
                    raise ValueError("Invalid Y-axis in saved selections")
               
                if selections['plot_type'] in plot_dropdown['values']:
                    plot_var.set(selections['plot_type'])
                else:
                    raise ValueError("Invalid plot type in saved selections")
               
                if selections['primary_sort'] in sort_dropdown['values']:
                    sort_var.set(selections['primary_sort'])
                else:
                    raise ValueError("Invalid primary sort in saved selections")
               
                if selections['secondary_sort'] in secondary_sort_dropdown['values']:
                    secondary_sort_var.set(selections['secondary_sort'])
                else:
                    raise ValueError("Invalid secondary sort in saved selections")
               
                if selections['sort_order'] in order_dropdown['values']:
                    order_var.set(selections['sort_order'])
                else:
                    raise ValueError("Invalid sort order in saved selections")
               
                print("Selections restored successfully")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error restoring selections: {str(e)}")
                print("Skipping restore due to incompatible or corrupted selections file")
        else:
            print("No saved selections found")
 
    # restore_button = ttk.Button(root, text="Restore Selections", command=restore_selections)
    # restore_button.grid(row=max_rows+3, column=0, columnspan=num_columns*2, pady=10)
 
    def reset_selections():
        # Reset dropdown selections
        for column, var in dropdown_vars:
            var.set("N/A")
       
        # Reset X-axis, Y-axis, and plot type
        x_var.set("Index")
        y_var.set("")  # Empty string as there's no default Y-axis
        plot_var.set("Scatter")
       
        # Reset sorting options
        sort_var.set("None")
        secondary_sort_var.set("None")
        order_var.set("Ascending")
       
        print("All selections have been reset to default values")
 
    reset_button = ttk.Button(root, text="Reset Selections", command=reset_selections)
    reset_button.grid(row=max_rows+4, column=0, columnspan=num_columns*2, pady=10)
 
    def on_closing():
        plt.close('all')  # Close all open plots
        root.destroy()    # Close the main GUI window
 
    root.protocol("WM_DELETE_WINDOW", on_closing)  # Bind the closing event
   
    # Attempt to restore selections when the GUI is created
    restore_selections()
   
    root.mainloop()
 
def filter_and_plot(df, selected_values, x_axis, y_axis, plot_type,
                    primary_sort, secondary_sort, sort_order, color):
    # Filter the dataframe based on selected values
    filtered_df = df.copy()
    legend_items = []
 
    # Create a boolean mask for filtering
    mask = pd.Series([True] * len(filtered_df))
    for column, value in selected_values.items():
        if value != "N/A":
            mask &= (filtered_df[column].astype(str) == value)
            legend_items.append(f"{column}: {value}")
   
    # Apply the mask to filter the dataframe
    filtered_df = filtered_df[mask]
 
    # Sort the dataframe
    ascending = True if sort_order == "Ascending" else False
    if primary_sort != "None":
        if secondary_sort != "None":
            filtered_df = filtered_df.sort_values(by=[primary_sort, secondary_sort], ascending=[ascending, ascending])
        else:
            filtered_df = filtered_df.sort_values(by=primary_sort, ascending=ascending)
 
    # Reset index to create a new sequential index
    filtered_df = filtered_df.reset_index(drop=True)
   
    if x_axis == "Index":
        x_data = filtered_df.index
        plt.xlabel("Index")
    elif x_axis == "GEMM_SIZE":
        x_data=filtered_df["M"].astype(str) + '_' +filtered_df["N"].astype(str) + '_' + filtered_df["K"].astype(str)
        plt.xlabel("GEMM_SIZE")
    else:
        x_data = filtered_df[x_axis]
        plt.xlabel(x_axis)
 
 
    y_data = filtered_df[y_axis]
   
    if plot_type == "Scatter":
        handle = plt.scatter(x_data, y_data, color=color)
    elif plot_type == "Line":
        handle, = plt.plot(x_data, y_data, color=color)
    elif plot_type == "Bar":
        handle = plt.bar(x_data, y_data, color=color)
        handle = handle[0]  # Use the first bar as the handle for the legend
    elif plot_type == "Dash":
        handle, = plt.plot(x_data, y_data, color=color, linestyle='--')
    elif plot_type == "Step":
        handle, = plt.step(x_data, y_data, color=color, where='mid')
    elif plot_type == "Stem":
        handle = plt.stem(x_data, y_data, linefmt=color, markerfmt=f'{color}o')
        handle = handle[0]  # Use the stem line as the handle for the legend
    elif plot_type == "Area":
        handle = plt.fill_between(x_data, y_data, color=color, alpha=0.5)
    elif plot_type == "Filled Line":
        handle, = plt.plot(x_data, y_data, color=color)
        plt.fill_between(x_data, y_data, alpha=0.3, color=color)
    elif plot_type == "Horizontal Bar":
        handle = plt.barh(x_data, y_data, color=color)
        handle = handle[0]  # Use the first bar as the handle for the legend
        plt.xlabel, plt.ylabel = plt.ylabel, plt.xlabel  # Swap x and y labels
    elif plot_type == "Box Plot":
        handle = plt.boxplot([y_data], positions=[0], patch_artist=True)
        for patch in handle['boxes']:
            patch.set_facecolor(color)
        handle = handle['boxes'][0]  # Use the box as the handle for the legend
        plt.xticks([0], [''])  # Remove x-axis label
    elif plot_type == "Violin Plot":
        handle = plt.violinplot([y_data], positions=[0], showmeans=True, showextrema=True, showmedians=True)
        for pc in handle['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
        handle = handle['bodies'][0]  # Use the violin body as the handle for the legend
        plt.xticks([0], [''])  # Remove x-axis label
 
    # Adjust x-axis for categorical data
    if x_axis != "Index" and not np.issubdtype(x_data.dtype, np.number):
        plt.xticks(range(len(x_data)), x_data, rotation=45, ha='right')
 
   
    plt.ylabel(y_axis)
    title = f"{y_axis} vs {x_axis}"
    if primary_sort != "None":
        title += f"\nSorted by {primary_sort}"
        if secondary_sort != "None":
            title += f", then by {secondary_sort}"
        title += f" ({sort_order})"
    plt.title(title)
 
    # Set x-axis ticks
    num_ticks = 15
    if len(x_data) > num_ticks:
        step = max(1, len(x_data) // (num_ticks - 1))
        tick_positions = np.arange(0, len(x_data), step)
        if len(tick_positions) > num_ticks:
            tick_positions = tick_positions[:num_ticks]
        elif len(tick_positions) < num_ticks and len(x_data) > num_ticks:
            tick_positions = np.append(tick_positions, len(x_data) - 1)
       
        plt.xticks(tick_positions, [x_data[i] if i < len(x_data) else '' for i in tick_positions])
    else:
        plt.xticks(range(len(x_data)), x_data)
 
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
 
    label = f"{y_axis} ({', '.join(legend_items)})"
    return {'handle': handle, 'label': label}
 
def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <csv_file_path>")
        sys.exit(1)
 
    csv_file_path = sys.argv[1]
   
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file_path}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Unable to parse '{csv_file_path}'. Make sure it's a valid CSV file.")
        sys.exit(1)
 
    unique_values = get_unique_values(csv_file_path)
    create_gui(unique_values, df)
 
if __name__ == "__main__":
    main()
