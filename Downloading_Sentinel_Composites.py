# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:33:17 2024

@author: Navid Mahdizadeh
"""

# Importing necessary libraries/modules
import datetime

# Mounting Google Drive
from google.colab import drive
drive.mount("/content/drive")

# Connecting to Google Earth Engine (GEE) through Earth Engine Python API
import ee
ee.Authenticate()
ee.Initialize(project='ee-navidmehdizade73nm')

# Setting the period for downloading the Sentinel-2 satellite images
def prompt_for_date(prompt_message):
    while True:
        date_str = input(prompt_message)
        try:
            # Try to create a datetime object from the input
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            # If there is a ValueError, it means the format is incorrect
            print("Incorrect date format, should be YYYY-MM-DD. Please try again.")

# Prompting the user for start and end dates
start_date_str = prompt_for_date("Enter the start date (YYYY-MM-DD): ") # e.g., 2021-04-01
end_date_str = prompt_for_date("Enter the end date (YYYY-MM-DD): ") # e.g., 2021-10-31

start_date = ee.Date(start_date_str)
end_date = ee.Date(end_date_str)

# Calculating number of 10-day intervals in the given period for downloading the Sentinel-2
number_of_intervals = end_date.difference(start_date, 'day').divide(10).ceil().getInfo()
print('number_of_intervals: ', number_of_intervals)


# Extracting bands' values of the Sentinel-2 satellite imagery across the study area
# Initializing shapefiles after uploading the study area into assets of the GEE
polygon_shapefile = ee.FeatureCollection('projects/ee-navidmehdizade73nm/assets/Crop_Boundary_CNN')


# Function to process a specific interval
def process_interval(index, start_date, end_date, region):
    interval = 10  # days
    offset = start_date.advance(interval * index, 'day')
    interval_start = offset
    interval_end = offset.advance(interval, 'day')

    if interval_end.difference(end_date, 'day').gt(0):
        interval_end = end_date

    # Sentinel-2 image collection
    image_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(interval_start, interval_end) \
        .filterBounds(region) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    # Select bands and rename
    renamed_collection = image_collection.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8', 'B8A', 'B11','B12'], ['Blue', 'Green', 'Red', 'Red_edge_1', 'Red_edge_2','Red_edge_3','NIR', 'Red_edge_4','SWIR_1','SWIR_2'])

    # Reduce the collection to a single image (i.e., composite) by taking the median values
    median_values_image = renamed_collection.reduce(ee.Reducer.median())

    # Export the single image as GeoTIFF
    task = ee.batch.Export.image.toDrive(
        image=median_values_image,
        description=f'Composite_{index}',
        folder='Sentinel2_median_composites',
        region=region,  # Use the region of the raster file
        scale=10,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f'Processed interval {index}')

# Process each interval
for ind in range(int(number_of_intervals)):
    process_interval(ind, start_date, end_date, polygon_shapefile)

print('Pay attention: Although it appears that the code has finished executing, the GeoTIFF files will take some time to appear in your Google Drive.')
