import sys
import pandas as pd
import math

def degree_to_radius(degree):
    return degree * math.pi / 180

# The following function is adapted from https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points , similar to the Exercise 3.
def distance(city, stations):
    '''
    Given a city and all stations, calculate the distance of every city-station pair
    Using 2D GPS
    '''
    lon_diff_rad = degree_to_radius(city.longitude - stations.longitude)
    lat_diff_rad = degree_to_radius(city.latitude - stations.latitude)
    lat_city_rad = degree_to_radius(city.latitude)
    lat_stations_rad = degree_to_radius(stations.latitude)
    temp = (lat_diff_rad/2).apply(math.sin) ** 2 + (lon_diff_rad/2).apply(math.sin) ** 2 * math.cos(lat_city_rad) * (lat_stations_rad.apply(math.cos))
    distances = 2 * ((temp.apply(math.sqrt) / (1-temp).apply(math.sqrt)).apply(math.atan))
    return 6371 * distances

def best_tmax(city, stations):
    distances = distance(city, stations)
    # To get the station which is nearest to the city.
    station_best = stations.iloc[distances.idxmin()]
    return station_best.avg_tmax


if __name__ == '__main__':
	# To check the command line arguments.
    if len(sys.argv) != 4:
    	print(len(sys.argv))
    	print("Wrong number of arguments! Usage: python3 average_ratings.py movie_list.txt movie_ratings.csv output.csv")
    	sys.exit()

    # Parse the command line arguments.
    output_filename = sys.argv[-1]
    station_filename = sys.argv[1]
    city_filename = sys.argv[2]

    # Read data.
    station_df = pd.read_json(station_filename, lines=True)
    city_df = pd.read_csv(city_filename, header=0)

    # Data preprocessing.
    station_df['avg_tmax'] /= 10
    city_df.dropna(axis=0, inplace=True)
    city_df['area'] /= 10**6
    city_df.drop(city_df[city_df['area'] > 10000].index, inplace=True)
    city_df['population density'] = city_df['population'] / city_df['area']

    # Calculate the best average max temperature for each city.
    city_df['avg_tmax'] = city_df.apply(best_tmax, axis=1 , stations=station_df)

    # Data visualization.
    scatter_plot = city_df.plot.scatter(x='avg_tmax',
                                        y='population density',
                                        c='b')
    scatter_plot.set_title('Temperature vs Population Density')
    scatter_plot.set_xlabel('Avg Max Temperature (\u00b0C)')
    scatter_plot.set_ylabel('Population Density (people/km\u00b2)')
    scatter_plot.figure.savefig(output_filename)

