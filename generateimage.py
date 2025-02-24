import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Ensures plots display correctly in an interactive environment
import matplotlib.pyplot as plt
import re
from ast import literal_eval

# Load the CSV file
data = pd.read_csv('jwst_dataset_10.csv')

# Preprocess the 'image' column to make it evaluable
def preprocess_image_column(image_str):
    """
    Preprocess the 'image' column string by removing 'array(...)' and 'dtype=...' 
    to simplify parsing with eval().
    
    Args:
        image_str (str): The string representation of the image data.
    
    Returns:
        str: The cleaned string ready for evaluation.
    """
    # image_str = re.sub(r'array\((\[.*?\])\)', r'\1', image_str)  # Remove 'array([...])' wrappers
    # image_str = re.sub(r',\s*dtype=[a-zA-Z0-9_]+', '', image_str)  # Remove dtype specifiers
   
    # image_str = image_str.replace('array', 'np.array')  # Remove any remaining 'array' keywords
    # image_str = re.sub(r'array\((\[.*?\])\)', r'np.array(\1)', image_str)  # Replace 'array([...])' with 'np.array([...])'

    # 1️⃣ Remove dtype specifiers first
    image_str = re.sub(r',\s*dtype=[a-zA-Z0-9_]+', '', image_str)

    # 2️⃣ Recursively remove all array(...) wrappers
    while 'array(' in image_str:
        image_str = re.sub(r'array\(([^()]+)\)', r'\1', image_str)  # Only replace innermost array()

    return image_str
    return image_str

data['image'] = data['image'].apply(preprocess_image_column)

# Extract flux data for each band
def extract_image_data(row):
    """
    Extract flux data for each band and convert it into 96x96 NumPy arrays.
    Handles cases where flux data might be nested lists or string representations.
    
    Args:
        row (pd.Series): A row from the DataFrame containing the 'image' column.
    
    Returns:
        pd.Series: A series with 'bands' (list of band names) and 'fluxes' (list of 96x96 arrays).
    """
    try:
        # Evaluate the preprocessed string into a dictionary
        image_data = eval(row['image'])
        bands = image_data['band']  # List of band names (e.g., ['f090w', 'f150w'])
        fluxes = image_data['flux']  # List of flux data for each band
        extracted_fluxes = []

        for i, band_flux in enumerate(fluxes):
            if isinstance(band_flux, list):
                # Check if the list elements are strings (e.g., stringified lists)
                if all(isinstance(row, str) for row in band_flux):
                    try:
                        # Parse each string into a list
                        band_flux = [literal_eval(row) for row in band_flux]
                    except Exception as e:
                        print(f"Band {bands[i]}: Failed to parse inner strings ({e}), using zeros")
                        band_flux = [np.zeros(96)] * 96  # Fallback to zeros
                elif len(band_flux) == 9216 and all(isinstance(x, (int, float)) for x in band_flux):
                    # Handle case where flux is a flat list of 9216 elements
                    flux_array = np.array(band_flux).reshape(96, 96)
                    extracted_fluxes.append(flux_array)
                    print(f"Band {bands[i]}: Reshaped 9216-element list to 96x96")
                    continue

                # Convert to NumPy array and validate shape
                flux_array = np.array(band_flux)
                if flux_array.shape == (96, 96):
                    print(f"Band {bands[i]}: Found 96x96 image")
                    extracted_fluxes.append(flux_array)
                else:
                    print(f"Band {bands[i]}: Unexpected shape {flux_array.shape}, using zeros")
                    extracted_fluxes.append(np.zeros((96, 96)))
            else:
                print(f"Band {bands[i]}: Flux is not a list, using zeros")
                extracted_fluxes.append(np.zeros((96, 96)))

        return pd.Series({'bands': bands, 'fluxes': extracted_fluxes})
    except Exception as e:
        print(f"Error processing row: {e}")
        return pd.Series({'bands': [], 'fluxes': []})

# Apply extraction to all rows
expanded_data = data.apply(extract_image_data, axis=1)
data = pd.concat([data, expanded_data], axis=1)

# Plot the images
def plot_images(row, index):
    """
    Plot the images for each band of a galaxy, handling negative values and enhancing visibility.
    
    Args:
        row (pd.Series): A row from the DataFrame with 'bands' and 'fluxes'.
        index (int): The index of the row for labeling purposes.
    """
    bands = row['bands']
    images = row['fluxes']
    if not bands or not images:
        print(f"Galaxy {index}: No data to plot")
        return

    fig, axes = plt.subplots(1, len(bands), figsize=(15, 5))
    fig.suptitle(f'Galaxy {index} - Multi-band Images', fontsize=16)

    # Handle single-band case
    axes = [axes] if len(bands) == 1 else axes

    for i, (band, img) in enumerate(zip(bands, images)):
        ax = axes[i]
        # Shift negative values to non-negative
        img = img - np.min(img) if np.min(img) < 0 else img
        # Normalize to [0, 1]
        if np.max(img) > 0:
            img = img / np.max(img)
        # Enhance faint features with logarithmic scaling
        img_display = np.log1p(img * 1000)  # Adjust multiplier if needed
        ax.imshow(img_display, cmap='inferno', origin='lower')
        ax.set_title(f'Band: {band} ({img.shape})')
        ax.axis('off')
        print(f"Band {band} min/max after normalization: {img.min():.6f}, {img.max():.6f}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


    

# Plot images for the first 5 galaxies
for i in range(min(5, len(data))):
    plot_images(data.iloc[i], index=i)