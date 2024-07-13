import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.cluster import hierarchy
import argparse


print("Pandas version:", pd.__version__)
print("Seaborn version:", sns.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Scipy version:", scipy.__version__)

def load_and_process_data(csv_file):
    df = pd.read_csv(csv_file, header=None)
    df1 = df[0].str.split(';', expand=True)
    df = df.drop(0, axis=1)
    merged_df = pd.concat([df1, df], axis=1)
    processed_df = merged_df.set_index(merged_df[0]).T.reset_index(drop=True).rename_axis(None, axis=1)
    processed_df = processed_df.drop(index=0).reset_index(drop=True)
    processed_df.index = processed_df.columns
    print(processed_df.shape)
    return processed_df


def plot_clustermap(data, output_file):
    # Performing hierarchical clustering for rows and columns
    row_linkage = hierarchy.linkage(data.astype(float).values, method='average')
    col_linkage = hierarchy.linkage(data.astype(float).values.T, method='average')

    # Creating a clustered heatmap
    plt.figure(figsize=(10, 8))
    clustered_heatmap = sns.clustermap(
        data.astype(float),
        # cmap = 'viridis',
        row_cluster=True,
        col_cluster=True,
        row_linkage=row_linkage,
        col_linkage=col_linkage
    )

    clustered_heatmap.savefig(output_file)
    plt.close()

    # Getting the order of rows after clustering
    row_order = clustered_heatmap.dendrogram_row.reordered_ind
    row_names = data.index[row_order]

    return row_names


def save_row_names(row_names, output_csv):
    # Writing the row names to a CSV file
    row_names_df = pd.DataFrame({'Drugs': row_names})
    row_names_df.to_csv(output_csv, index=False)


def main(csv_file, plot_file, output_csv):
    data = load_and_process_data(csv_file)
    row_names = plot_clustermap(data, plot_file)
    save_row_names(row_names, output_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("plot heatmap clusters")
    parser.add_argument('-p','--input_path', type=str, required=True, help="Generalized CSV input path")
    args = parser.parse_args()
    main(args.input_path, 'clustermap.png', 'clustermap.csv')
