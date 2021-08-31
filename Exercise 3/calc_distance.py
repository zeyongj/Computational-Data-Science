from xml.dom import minidom
import sys
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

def get_data(filename):
    # Adapted from https://stackabuse.com/reading-and-writing-xml-files-in-python .
    mydoc = minidom.parse(sys.argv[1])
    items = mydoc.getElementsByTagName('trkpt')
    location = ['lat','lon']
    dataFrame = pd.DataFrame(columns = location)
    for item in items:
        lat = item.getAttribute('lat')
        latitude = float(lat)
        lon = item.getAttribute('lon')
        longitude = float(lon)
        df = {"lat": latitude, "lon": longitude}
        dataFrame = dataFrame.append(df, ignore_index = True)
    return dataFrame

# The following function is adapted from https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points .
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers.
    distance_in_meter = c * r * 1000
    return distance_in_meter

def distance(place):
    df = place.copy()
# The following codes are adapted from https://stackoverflow.com/questions/53697724/getting-distance-from-longitude-and-latitude-using-haversines-distance-formula .
    df['latitude'] = df['lat'].shift(1)
    df['longitude'] = df['lon'].shift(1)
    df['distance'] = df.apply(lambda row: haversine(row['lat'], row['lon'], row['latitude'], row['longitude']), axis = 1)
    distance_in_meter = df['distance'].sum()
    return distance_in_meter

def smooth(place):
    kalman_data = place
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([0.20, 0.20]) ** 2 # TODO: shouldn't be zero
    transition_covariance = np.diag([0.00001, 0.00001]) ** 2 # TODO: shouldn't be zero
    transition = [[1,0], [0,1]] # TODO: shouldn't (all) be zero. Adapted from https://stackoverflow.com/questions/55195011/how-to-define-state-transition-matrix-for-kalman-filters: DT = np.matrix([[1.,0.,dt,0],[0.,1.,0.,dt],[0.,0.,1.,0.],[0.,0.,0.,1.]]).
    kf = KalmanFilter(initial_state_mean = initial_state,
                      observation_covariance = observation_covariance,
                      transition_covariance = transition_covariance,
                      transition_matrices = transition)
    kalman_smoothed, _ = kf.smooth(kalman_data)
    location = ['lat','lon']
    answer = pd.DataFrame(data = kalman_smoothed, columns = location)
    return answer

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def main():
    points = get_data(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points),))
    
    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()
