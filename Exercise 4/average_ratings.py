import sys
import pandas as pd
import numpy as np
from difflib import get_close_matches


def get_close_matches_wrap(title):
    # The routine to find the best-match real movie title, if none found, return NaN
    # First, get the nearest neighbor list.
    title_nn_list = get_close_matches(title, movie_df['title'], n=1, cutoff=0.6)
    # Then, check whether the list is empty.
    if title_nn_list:
        return title_nn_list[0]
    else:
        return np.nan


if __name__ == '__main__':
	# To check the command line arguments.
    if len(sys.argv) != 4:
    	print(len(sys.argv))
    	print("Wrong number of arguments! Usage: python3 average_ratings.py movie_list.txt movie_ratings.csv output.csv")
    	sys.exit()

    # Parse the command line arguments.
    output_filename = sys.argv[-1]
    movie_list_filename = sys.argv[1]
    movie_ratings_filename = sys.argv[2]

    # Read data.
    movie_df = pd.read_csv(movie_list_filename, sep='\n', header=None, names=['title'])
    movie_rating_df = pd.read_csv(movie_ratings_filename, header=0)

    # Add a column to the movie rating table, the best-match real movie title.
    movie_rating_df['real title'] = movie_rating_df['title'].apply(get_close_matches_wrap)
    # Drop rows without a valid movie title.
    movie_rating_valid_df = movie_rating_df[['rating','real title']].dropna(axis=0, inplace=False).rename(columns={'real title':'title'})

    # Group by movie titles and calculate the mean rating for each group, sort by movie title.
    movie_rating_mean = movie_rating_valid_df.groupby(by='title', sort=True)['rating'].mean()
    movie_rating_mean.round(2).to_csv(output_filename)
