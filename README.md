# Objectives
The goals of this challenge are for you to:

Create an automated ETL pipeline.
Extract data from multiple sources.
Clean and transform the data automatically using Pandas and regular expressions.
Load new data into PostgreSQL.

# Challenge Analysis

Streamlined 3 different functions into one cohesive chain merging various datasets:

Wikipedia data,
Kaggle metadata,
MovieLens rating data (from Kaggle)

Performed ETL to clean and transform each datasets to get rid of dirty data points ie removed null or duplicate rows, ensured correct syntax formats for the columns and row, created functions to further clean the data, used regex to find and replace data

The challange.py has been refactored and cleaned to optimize the combination of the 3 different datasets.  All misc commentary and functions are improved (adding try/except blocks, parsing out information, cleaning column names).  There is now also an automated process that loads all 3 cleaned datasets for Movies, Movie Ratings and the Ratings table.
The ETL process was completed top ensure that these three tablesets are easy to merge and be queried as needed.
