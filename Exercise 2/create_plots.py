import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Data Preprocessing:
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    first_dataset = pd.read_csv(filename1, sep=' ', header=None, index_col=1,names=['lang', 'page', 'views', 'bytes'])
    # Adapted from https://coursys.sfu.ca/2021su-cmpt-353-d1/pages/Exercise2
    sorted_first_dataset = first_dataset.sort_values(by = "views",ascending = False)
    # Adapted from https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html

    # Data Visualization:
    # Plot 1: Distribution of Views
    plt.figure(figsize=(10, 5)) # change the size to something sensible: I don't change.
    plt.subplot(1, 2, 1) # subplots in 1 row, 2 columns, select the first
    plt.plot(sorted_first_dataset['views'].values) # build plot 1
    plt.title('Popularity Distribution')
    plt.xlabel('Rank')
    plt.ylabel('Views')
    # Adapted from https://coursys.sfu.ca/2021su-cmpt-353-d1/pages/Exercise2


    # Data Preprocessing:
    second_dataset = pd.read_csv(filename2, sep=' ', header=None, index_col=1,names=['lang', 'page', 'views', 'bytes'])
    merged_second_dataset = first_dataset.merge(second_dataset, left_on='page', right_on='page')
    # Adapted from https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html

    # Data Visualization:
    # Plot 2: Hourly Views
    plt.subplot(1, 2, 2) # ... and then select the second
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Hour 1 Views')
    plt.ylabel('Hour 2 Views')
    plt.title('Hourly Correlation')
    
    plt.scatter(merged_second_dataset['views_x'], merged_second_dataset['views_y'], c = 'b', s = 10)
    # Adapted from https://moonbooks.org/Articles/How-to-increase-the-size-of-scatter-points-in-matplotlib-/
    plt.savefig('wikipedia.png')

if __name__ == '__main__':
    main()
    
