#!/usr/bin/env python

################################################################################
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys


################################################################################
def latlon_to_cartesian(lats, lons, depths, ref_lat, ref_lon):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(ref_lat))

    x = (lons - ref_lon) * km_per_deg_lon
    y = (lats - ref_lat) * km_per_deg_lat
    z = depths  

    return np.column_stack([x, y, z])



################################################################################
def cartesian_to_latlon(coords, ref_lat, ref_lon):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(ref_lat))

    lons = ref_lon + coords[:, 0] / km_per_deg_lon
    lats = ref_lat + coords[:, 1] / km_per_deg_lat
    depths = coords[:, 2]  

    return lats, lons, depths



################################################################################
def calc_faultplane_from_pca(components, eigenvalues):
    """
    Calclulate Strike and Dip of a fault plane obtained from PCA analysis
    """

    normal_idx = np.argmin(eigenvalues)
    normal_vector = components[normal_idx]

    if normal_vector[2] > 0:
        normal_vector = -normal_vector

    cvertical = abs(normal_vector[2]) / np.linalg.norm(normal_vector)
    dip = np.degrees(np.arccos(cvertical))

    hnormal = np.array([-normal_vector[1], normal_vector[0], 0])
    hnormal_length = np.linalg.norm(hnormal)

    if hnormal_length > 1e-10:  
        hnormal = hnormal / hnormal_length
        strike = np.degrees(np.arctan2(hnormal[0], hnormal[1]))
        if strike < 0:
            strike += 360
    else:
        strike = 0  

    return strike, dip, normal_vector


################################################################################
def perform_pca_analysis(df, confidence_factor = 2.0, verbose=False ):
    ref_lat = df['lat'].mean()
    ref_lon = df['lon'].mean()

    coords = latlon_to_cartesian(df['lat'].values, df['lon'].values,
                                df['depth_km'].values, ref_lat, ref_lon)

    if verbose:
        print(f"Data range")
        print(f"X (E-W): {coords[:, 0].min():.2f} ~ {coords[:, 0].max():.2f} km")
        print(f"Y (N-S): {coords[:, 1].min():.2f} ~ {coords[:, 1].max():.2f} km")
        print(f"Z (Depth): {coords[:, 2].min():.2f} ~ {coords[:, 2].max():.2f} km")

    pca = PCA(n_components=3)
    pca.fit(coords)

    components = pca.components_
    eigenvalues = pca.explained_variance_

    if verbose:
        print("\n=== PCA analysis ===")
        print(f"eigenvalues: {eigenvalues}")
        print(f"variance ratio: {pca.explained_variance_ratio_}")
        print(f"cumulative sum of variance: {np.cumsum(pca.explained_variance_ratio_)}")

        for i, (comp, eigval) in enumerate(zip(components, eigenvalues)):
            print(f"PC{i+1}: [{comp[0]:.4f}, {comp[1]:.4f}, {comp[2]:.4f}] (λ={eigval:.4f})")

    strike, dip, normal_vector = calc_faultplane_from_pca(components, eigenvalues)

    if verbose:
        print(f"\n=== Fault plane ===")
        print(f"Strike: {strike:.1f}°")
        print(f"Dip: {dip:.1f}°")

    largest_two_indices = np.argsort(eigenvalues)[-2:][::-1]  
    pc1_idx, pc2_idx = largest_two_indices

    strike_length = 2 * np.sqrt(eigenvalues[pc1_idx])  
    dip_length = 2 * np.sqrt(eigenvalues[pc2_idx])     

    if verbose:
        print(f"Length of Strike direction: {strike_length:.2f} km")
        print(f"Length of Dip direction: {dip_length:.2f} km")

    center = np.mean(coords, axis=0)
    center_lat, center_lon, center_depth = cartesian_to_latlon(center.reshape(1, -1), ref_lat, ref_lon)

    if verbose:
        print(f"\n=== Fault center ===")
        print(f" {center_lat[0]:.6f},   {center_lon[0]:.6f},    {center_depth[0]:.2f} km")


    vertices_local = np.array([
        [-confidence_factor * np.sqrt(eigenvalues[pc1_idx]), -confidence_factor * np.sqrt(eigenvalues[pc2_idx]), 0],
        [confidence_factor * np.sqrt(eigenvalues[pc1_idx]), -confidence_factor * np.sqrt(eigenvalues[pc2_idx]), 0],
        [confidence_factor * np.sqrt(eigenvalues[pc1_idx]), confidence_factor * np.sqrt(eigenvalues[pc2_idx]), 0],
        [-confidence_factor * np.sqrt(eigenvalues[pc1_idx]), confidence_factor * np.sqrt(eigenvalues[pc2_idx]), 0]
    ])

    rotation_matrix = components.T
    vertices_rotated = vertices_local @ rotation_matrix.T + center

    vertices_lat, vertices_lon, vertices_depth = cartesian_to_latlon(vertices_rotated, ref_lat, ref_lon)

    vertices_df = pd.DataFrame({
        'lat': vertices_lat,
        'lon': vertices_lon,
        'depth_km': vertices_depth
    })

    if verbose:
        print(f"\n=== Vertices of Fault plane ===")
        for i, row in vertices_df.iterrows():
            print(f"Vertex {i+1}: {row['lat']:.6f}, {row['lon']:.6f}, {row['depth_km']:.2f} km")

    return {
#        'pca': pca,
#        'components': components,
#        'eigenvalues': eigenvalues,
        'strike': strike,
        'dip': dip,
#        'normal_vector': normal_vector,
        'strike_length': strike_length,
        'dip_length': dip_length,
        'center': center,
        'center_lat': center_lat[0],
        'center_lon': center_lon[0],
        'center_depth': center_depth[0],
        'vertices_df': vertices_df,
        'ref_lat': ref_lat,
        'ref_lon': ref_lon,
        'coords': coords
    }

################################################################################
def plot_3d_with_plane( df, view_elev=20, view_azim=240, planedf=None,
    save_path="fault_3d_plot.png"):
    """
    Parameters:
        df: DataFrame with 'lat', 'lon', 'depth_km'
        view_elev: elevation angle
        view_azim: azimuth angle
        planedf: DataFrame with 4 points ('lat', 'lon', 'depth_km')
        save_path: path to save PNG
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(df["lon"], df["lat"], df["depth_km"],  # depth is downward positive
               c=df["depth_km"], cmap="viridis", s=20, alpha=0.8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth (km)")

    # Reverse z-axis (downward positive)
    ax.set_zlim(df["depth_km"].max(), df["depth_km"].min())

    # Set viewing angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Optional: add overlay plane
    if len(planedf) == 4:
        ax.set_zlim(planedf["depth_km"].max(), planedf["depth_km"].min())
        xs = planedf["lon"].values
        ys = planedf["lat"].values
        zs = planedf["depth_km"].values

        verts = [list(zip(xs, ys, zs))]

        poly = Poly3DCollection(verts, alpha=0.3, color="red")
        ax.add_collection3d(poly)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved 3D plot to {save_path}")
    plt.show()



################################################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fault Plane Estimation")

    parser.add_argument('--EVLOC', type=str, required=True, help='CSV file of event distribution')

    args = parser.parse_args()

#    df = pd.read_pickle("sample1_165_15.pkl")

    try:
        df = pd.read_csv(args.EVLOC)
        result = perform_pca_analysis(df, confidence_factor = 2.0, verbose=False )

        print(f"\nNumber of events: {len(df)}")
        print(f"\nStrike: {result['strike']:4.1f}   Dip: {result['dip']:4.1f}")
        print(f"Strike length: {result['strike_length']:4.1f}   Dip length: {result['dip_length']:4.1f}")
        print(f"\n=== Vertices of Fault plane ===")
#        print(result['vertices_df'])
        for i, row in result['vertices_df'].iterrows():
            print(f"Vertex {i+1}: {row['lat']:.6f}, {row['lon']:.6f}, {row['depth_km']:.2f} km")

        plot_3d_with_plane(df, planedf=result['vertices_df'])

        sys.exit(0)

    except FileNotFoundError:
        print(f"File not found: {args.EVLOC}")
    except pd.errors.EmptyDataError:
        print("empty CSV file")
    except pd.errors.ParserError:
        print("CSV parser error")
    except Exception as e:
        print("Unknown error:", e)


    sys.exit(-1)
