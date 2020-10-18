import geopandas as gpd
import shapely
from geovoronoi import voronoi_regions_from_coords
from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
from sklearn.cluster import KMeans, DBSCAN
import random
from shapely.geometry import Point, Polygon
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import logging

logging.basicConfig(filename='split_blocks.log', 
                    level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(name)s - %(levelname)s - %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.info("reading squares...")
squares = gpd.read_file("gis_data/Square_Boundaries-shp/")
logging.info("reading addresses...")
addresses = gpd.read_file("gis_data/Address_Points-shp/")
logging.info("filter addresses for residential..")
res_addresses = addresses[addresses.loc[:,'RES_TYPE'] == 'RESIDENTIAL'].copy()
del addresses

def generate_random(number, polygon):
    '''
    from https://gis.stackexchange.com/questions/207731/generating-random-coordinates-in-multipolygon-in-python
    modified to output a geopandas df
    '''
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
            
    points_df = gpd.GeoDataFrame(pd.DataFrame(range(len(points)),columns=['pt_id']),
                            crs="EPSG:4326",
                            geometry=points)
    return points_df

def add_random_pts(address_points_for_cluster):
    '''
    '''
    random_extent = 0.0001
    new_point_list = []
    for point in address_points_for_cluster:
        minx = point[0]-random_extent
        maxx = point[0]+random_extent
        miny = point[1]-random_extent
        maxy = point[1]+random_extent
        new_point_list.append((random.uniform(minx, maxx), random.uniform(miny, maxy)))
    address_points_plus_rand = np.vstack((address_points_for_cluster,np.array(new_point_list)))
    return address_points_plus_rand



def split_poly_into_equal_parts(poly, num_parts):
    '''
    '''
    points_df = generate_random(2500, poly)
    km = KMeans(n_clusters=num_parts)
    
    points_df.loc[:,'lat'] = points_df.loc[:,'geometry'].apply(lambda x: x.y)
    points_df.loc[:,'lon'] = points_df.loc[:,'geometry'].apply(lambda x: x.x)
    points_for_cluster = points_df.copy()
    points_for_cluster.drop(labels=['geometry','pt_id'],axis=1,inplace=True)
    
    kmcls = km.fit(points_for_cluster.values)
    
    points_w_cl = points_df.assign(cluster=kmcls.labels_)
    centers = kmcls.cluster_centers_
    
    centers_gseries = gpd.GeoSeries(map(Point, zip(centers[:,1], centers[:,0])))
    
    centroid_coords = np.array([coords for coords in (zip(centers[:,1], centers[:,0]))])
    
    poly_shapes, pts, poly_to_pt_assignments  = voronoi_regions_from_coords(centroid_coords, poly)
    
    poly_shapes_df = gpd.GeoDataFrame(pd.DataFrame(poly_to_pt_assignments,columns=['group']),
                            crs="EPSG:4326",
                            geometry=poly_shapes)
    
    return poly_shapes_df

def prep_addresses_for_cluster(address_pts, one_square_shape):
    '''
    '''
    address_within = address_pts[address_pts.within(one_square_shape)].copy()
    for_cluster = pd.get_dummies(address_within[['LATITUDE', 'LONGITUDE','STNAME']], prefix=['STNAME'])
    for_cluster.iloc[:,2:] = for_cluster.iloc[:,2:].multiply(500,axis=1)
    address_points_for_cluster = for_cluster.values
    return address_points_for_cluster
    
def find_address_clusters(address_points_for_cluster):
    '''
    '''
    km_silhouette = {}
    max_cluster = min(10,len(address_points_for_cluster))
#     print(f"max_num_clusters = {max_cluster-1}")
    for i in range(2,max_cluster):
        km = KMeans(n_clusters=i, random_state=0).fit(address_points_for_cluster)
        preds = km.predict(address_points_for_cluster)

        silhouette = silhouette_score(address_points_for_cluster,preds)
        km_silhouette[silhouette] = i
#         print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
#         print("-"*100)
    best_num_clusters = max(4,km_silhouette[max(km_silhouette.keys())])
    km = KMeans(n_clusters=best_num_clusters, random_state=0).fit(address_points_for_cluster)
    centers = km.cluster_centers_
    centroid_coords = np.array([coords for coords in (zip(centers[:,1], centers[:,0]))])
    centers_gseries = gpd.GeoSeries(map(Point, zip(centers[:,1], centers[:,0])))
#     address_pts_w_cluster = address_pts.assign(cluster=km.labels_)
    
    return centroid_coords

def split_on_poly_by_streetname(square_id):
    '''
    '''
    logging.info(f"working on shape_id: {square_id}")
    try:
        select_square = squares[squares.loc[:,'SQUARE']==square_id].copy()
        if (select_square.geometry.type=='MultiPolygon').any():
            select_square = select_square.explode()
        address_pts = addresses[addresses.loc[:,'SQUARE']==square_id].copy()
        square_part = 1
        for index, one_square in select_square.iterrows():
            one_square_shape = one_square['geometry']
            address_within = address_pts[address_pts.within(one_square_shape)].copy()
            if len(address_within)<4:
                split_type = "equal_area"
                poly_shapes_df = split_poly_into_equal_parts(one_square_shape, 4)
                poly_shapes_partial = gpd.sjoin(poly_shapes_df, address_within[['SQUARE','SSL','STNAME','geometry']], how='left')
                poly_shapes_partial.loc[:,'SQUARE_PART'] = square_part

                poly_shapes_partial = poly_shapes_partial[['group', 'geometry'
                                                        , 'SQUARE', 'SSL', 'STNAME','SQUARE_PART']]

            else:
                split_type = "streetname_breakdown"
                address_pts_array = np.array([coords for coords in address_within.geometry.apply(lambda x: (x.x,x.y))])
                poly_shapes, pts, poly_to_pt_assignments  = voronoi_regions_from_coords(address_pts_array
                                                                        , one_square_shape)

                poly_shapes_df = gpd.GeoDataFrame(pd.DataFrame(poly_to_pt_assignments,columns=['group']),
                        crs="EPSG:4326",
                        geometry=poly_shapes)

                poly_shapes_w_stname = gpd.sjoin(poly_shapes_df, address_within[['SQUARE','SSL','STNAME','geometry']])

                poly_shapes_partial = poly_shapes_w_stname.dissolve(by='STNAME').reset_index()
                    

                poly_shapes_partial.loc[:,'SQUARE_PART'] = square_part
                poly_shapes_partial = poly_shapes_partial[['group', 'geometry'
                                                        , 'SQUARE', 'SSL', 'STNAME','SQUARE_PART']]
                
            if square_part == 1:
                full_poly_shape_df = poly_shapes_partial.copy()
            else:
                full_poly_shape_df = full_poly_shape_df.append(poly_shapes_partial)
            square_part += 1
    except:
        bad_shape_df = pd.DataFrame([[0,Polygon([(0, 0), (1, 1), (0, 1)]),square_id,0]]
        , columns=['group', 'geometry','SQUARE','SQUARE_PART'])
        full_poly_shape_df = gpd.GeoDataFrame(bad_shape_df, crs="EPSG:4326", geometry='geometry')
        
    return full_poly_shape_df


def split_one_poly_into_parts(square_id):
    '''
    '''
    logging.info(f"working on shape_id: {square_id}")
    try:
        select_square = squares[squares.loc[:,'SQUARE']==square_id].copy()
        if (select_square.geometry.type=='MultiPolygon').any():
            select_square = select_square.explode()
        address_pts = addresses[addresses.loc[:,'SQUARE']==square_id].copy()
        square_part = 1
        for index, one_square in select_square.iterrows():
            one_square_shape = one_square['geometry']
            address_points_for_cluster = prep_addresses_for_cluster(address_pts, one_square_shape)
            if len(address_points_for_cluster)<4:
                split_type = "equal_area"
                poly_shapes_df = split_poly_into_equal_parts(one_square_shape, 4)
            else:
                split_type = "address_cluster"
                centroid_coords = find_address_clusters(address_points_for_cluster)

                poly_shapes, pts, poly_to_pt_assignments  = voronoi_regions_from_coords(centroid_coords, one_square_shape)

                poly_shapes_df = gpd.GeoDataFrame(pd.DataFrame(poly_to_pt_assignments,columns=['group']),
                                        crs="EPSG:4326",
                                        geometry=poly_shapes)
                    
                poly_shapes_df.loc[:,'SQUARE'] = square_id
                poly_shapes_df.loc[:,'SQUARE_PART'] = square_part
                
            if square_part == 1:
                full_poly_shape_df = poly_shapes_df.copy()
            else:
                full_poly_shape_df = full_poly_shape_df.append(poly_shapes_df)
            square_part += 1
    except:
        bad_shape_df = pd.DataFrame([[0,Polygon([(0, 0), (1, 1), (0, 1)]),square_id,'','',0]]
        , columns=['group', 'geometry','SQUARE','SSL','STNAME','SQUARE_PART'])

        full_poly_shape_df = gpd.GeoDataFrame(bad_shape_df, crs="EPSG:4326", geometry='geometry')
        
    return full_poly_shape_df

def split_one_poly_into_parts_forplotting(square_id):
    '''
    '''
    one_square = squares[squares.loc[:,'SQUARE']==square_id].copy()
    address_pts = addresses[addresses.loc[:,'SQUARE']==square_id].copy()
    address_points_for_cluster = address_pts[['LATITUDE', 'LONGITUDE']].values
    if len(address_points_for_cluster)<4:
        split_type = "equal_area"
        poly_shapes_df = split_poly_into_equal_parts(one_square.unary_union, 4)
    else:
        split_type = "address_cluster"
        km_silhouette = {}
        max_cluster = min(10,len(address_points_for_cluster))
    #     print(f"max_num_clusters = {max_cluster-1}")
        for i in range(2,max_cluster):
            km = KMeans(n_clusters=i, random_state=0).fit(address_points_for_cluster)
            preds = km.predict(address_points_for_cluster)

            silhouette = silhouette_score(address_points_for_cluster,preds)
            km_silhouette[silhouette] = i
    #         print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    #         print("-"*100)
        best_num_clusters = max(4,km_silhouette[max(km_silhouette.keys())])
        km = KMeans(n_clusters=best_num_clusters, random_state=0).fit(address_points_for_cluster)
        centers = km.cluster_centers_
        centroid_coords = np.array([coords for coords in (zip(centers[:,1], centers[:,0]))])
        centers_gseries = gpd.GeoSeries(map(Point, zip(centers[:,1], centers[:,0])))
        address_pts_w_cluster = address_pts.assign(cluster=km.labels_)
        centroid_filtered = centers_gseries[centers_gseries.apply(lambda x: one_square.unary_union.contains(x))].copy()
        centroid_coords_filtered = np.array([coords for coords in (zip(centroid_filtered.apply(lambda x: x.x), 
                                    centroid_filtered.apply(lambda x: x.y)))])
        
        poly_shapes, pts, poly_to_pt_assignments  = voronoi_regions_from_coords(centroid_coords_filtered, one_square.unary_union)

        poly_shapes_df = gpd.GeoDataFrame(pd.DataFrame(poly_to_pt_assignments,columns=['group']),
                                crs="EPSG:4326",
                                geometry=poly_shapes)
    
    poly_shapes_export = poly_shapes_df.assign(square=square_id)
        
    return (poly_shapes_export, address_pts, one_square)