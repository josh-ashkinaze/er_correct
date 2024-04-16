"""
Author: Joshua Ashkinaze

Description: Gets metadata for retracted articles.

INPUT DATA
I downloaded the retraction dataset crossref which recently acquired the Retraction Watch Database.
Specifically, on April 15 2024 I used the following link (which automatically downloaded the CSV at the time)
DL link: https://api.labs.crossref.org/data/retractionwatch?ginny@crossref.org

I found the link at the following page
DL link page: https://www.crossref.org/blog/news-crossref-and-retraction-watch/

PROCESSING
To get metadata I use OpenCitations. The goal is to get a before and after count of citations for each retracted article.


Date: 2024-04-15 18:54:58
"""

import pandas as pd
import requests
from datetime import datetime
import chardet
import logging
import os
import numpy as np
import time
from random import random


def standardize_date(date_str):
    if pd.isna(date_str):
        return np.nan
    parts = date_str.split('-')
    if len(parts) == 1:  # Only year is present
        return np.nan
    elif len(parts) == 2:  # Year and month are present
        return date_str + '-01'
    elif len(parts) == 3:  # Full date is present
        return date_str
    return np.nan  # For any other case that doesn't match expected patterns


def count_citations(citations, target_date):
    before_count = 0
    after_count = 0
    target_date = pd.to_datetime(target_date)

    for citation in citations:
        creation_date = standardize_date(citation['creation'])
        if pd.isna(creation_date):
            continue
        creation_date = pd.to_datetime(creation_date)
        if creation_date < target_date:
            before_count += 1
        elif creation_date > target_date:
            after_count += 1

    return {'before': before_count, 'after': after_count}


def fetch_citations(doi):
    """Fetch all citations for a given PMID using the OpenCitations API."""
    url = f"https://opencitations.net/index/api/v2/citations/doi:{doi}"
    headers = {"Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            logging.info(f"Successfully fetched citations for DOI {doi}")
            return response.json()
        else:
            return []
    except Exception as e:
        logging.info(f"Exception when fetching citations for DOI {doi}: {e}")
        return []


def process_citation_counts(citations, retraction_date):
    """Count citations before and after the retraction date using NumPy for efficient processing."""
    try:
        # Ensure retraction_date is a datetime.date object
        if isinstance(retraction_date, datetime):
            retraction_date = retraction_date.date()  # Convert datetime to date if necessary

        # Prepare an array of citation creation dates
        creation_dates = np.array(
            [datetime.strptime(citation['creation'], '%Y-%m-%d').date() for citation in citations])

        # Compare all dates in a vectorized manner
        pre_counts = np.sum(creation_dates < retraction_date)
        post_counts = np.sum(creation_dates >= retraction_date)

        return pre_counts, post_counts

    except Exception as e:
        print("ERROR:", e)
        return np.NaN, np.NaN


def read_csv_robust(file_path, sep=",", num_bytes=10000):
    # Detect the file encoding
    def detect_encoding(file_path, num_bytes):
        with open(file_path, 'rb') as file:
            rawdata = file.read(num_bytes)
            result = chardet.detect(rawdata)
            return result['encoding']

    encoding_detected = detect_encoding(file_path, num_bytes)

    # Try reading the file with the detected encoding
    try:
        df = pd.read_csv(file_path, encoding=encoding_detected, on_bad_lines='skip', sep=sep)
        print(f"File read successfully with encoding: {encoding_detected}")
        return df
    except Exception as e:
        print(f"Failed to read with detected encoding {encoding_detected}. Error: {e}")

        # Fallback to UTF-8
        try:
            print("Attempting to read with UTF-8...")
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', sep=sep)
            print("File read successfully with UTF-8.")
            return df
        except Exception as e:
            print(f"Failed to read with UTF-8. Error: {e}")

            # Second fallback to ISO-8859-1
            try:
                print("Attempting to read with ISO-8859-1...")
                df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip', sep=sep)
                print("File read successfully with ISO-8859-1.")
                return df
            except Exception as e:
                print(f"Failed to read with ISO-8859-1. Error: {e}")
                print("All attempts failed. Please check the file for issues beyond encoding.")
                return None


def main():
    logging.basicConfig(filename=f"{os.path.splitext(os.path.basename(__file__))[0]}.log", level=logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d__%H--%M--%S', filemode='w')
    logging.info("Starting up")
    ####################
    # Read in file and handle weird encoding
    ####################
    df = read_csv_robust("retractions.csv")  # since encoding was messed up when I downloaded this
    total_n = len(df)
    logging.info(f"Total number of rows: {total_n}")

    ####################
    # Fix the date columns
    ####################
    df['original'] = pd.to_datetime(df['OriginalPaperDate'], format='mixed', errors='coerce').dt.date
    df['retract'] = pd.to_datetime(df['RetractionDate'], format='mixed', errors='coerce').dt.date
    df['correction_time'] = (pd.to_datetime(df['retract']) - pd.to_datetime(df['original'])).dt.days
    df = df.dropna(subset=['correction_time'])
    logging.info(f"Number of rows after dropping correction time NaNs: {len(df)}")

    df = df.dropna(subset=['OriginalPaperDOI'])
    logging.info(f"Number of rows after dropping DOI NaNs: {len(df)}")

    ####################
    # Get citations
    ####################
    cites = []
    log_every = 50
    sleep_every = 50
    total_n = len(df)
    counter = 0
    dois = df['OriginalPaperDOI'].tolist()
    for i in range(len(dois)):
        cites.append(fetch_citations(dois[i]))
        counter+=1
        if counter % log_every == 0:
            logging.info(f"Processed {counter} of {total_n} rows")
        if counter % sleep_every == 0:
            time.sleep(random())
    df['cites'] = [i for i in cites]

    logging.info("Collected all citations")

    ####################
    # Get citations
    ####################
    df['citation_counts'] = df.apply(lambda row: count_citations(row['cites'], row['retract']), axis=1)

    df['before_count'] = df['citation_counts'].apply(lambda x: x['before'])
    df['after_count'] = df['citation_counts'].apply(lambda x: x['after'])
    df.drop(columns=['citation_counts'], inplace=True)

    logging.info("Finished processing")
    df.to_json("retractions_meta.json", lines=True, index=False, orient='records')
    logging.info("wrote file")


if __name__ == "__main__":
    main()
