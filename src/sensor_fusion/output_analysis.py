import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lidar_output_data_path = "./data/lidar_ekf_output.txt"
radar_output_data_path = "./data/radar_ekf_output.txt"


def plot_stats(data_path, sensor_type, save_dir="./images"):
    data_ = open(data_path, "r")
    output_list = []
    for num, line in enumerate(data_):
        line = line.rstrip("\n")

        if num == 0:
            headings = line.split(",")
        else:
            output_list.append(line.split(","))

    output_list = np.array(output_list, dtype=np.float32)
    output_pd = pd.DataFrame(output_list, columns=headings)

    # if sensor_type == "radar":
    output_pd = output_pd.drop(columns=["px_gt", "py_gt", "vx_gt", "vy_gt", "px_rmse", "py_rmse", "vx_rmse", "vy_rmse"], axis=1)

    output_pd.plot(figsize=(20, 10))
    path_ = os.path.join(save_dir, f"{sensor_type}.png")
    plt.savefig(path_)


if __name__ == "__main__":
    plot_stats(lidar_output_data_path, sensor_type="fusion_lidar", save_dir="./images")
    plot_stats(radar_output_data_path, sensor_type="fusion_radar", save_dir="./images")
