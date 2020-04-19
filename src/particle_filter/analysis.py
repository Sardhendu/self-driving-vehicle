import os
import numpy as np
import pandas as pd
from src import commons
import matplotlib.pyplot as plt

gt_prediction_data = "./data/gt_prediction.txt"


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
    output_pd = output_pd.drop(columns=["pr_id"], axis=1)

    print(output_pd.columns)

    pos_x = output_pd[["gt_x", "pr_x"]]
    pos_y = output_pd[["gt_y", "pr_y"]]
    yaw_theta = output_pd[["gt_theta", "pr_theta"]]
    particle_weight = output_pd[["pr_weight", "pr_avg_weight"]]

    error_x = abs(np.array(list(pos_x["gt_x"])) - np.array(list(pos_x["pr_x"])))
    pos_x["error_x"] = error_x

    error_y = abs(np.array(list(pos_y["gt_y"])) - np.array(list(pos_y["pr_y"])))
    pos_y["error_y"] = error_y

    error_theta = abs(np.array(list(yaw_theta["gt_theta"])) - np.array(list(yaw_theta["pr_theta"])))
    yaw_theta["error_theta"] = error_theta

    error_xy = pd.DataFrame(pow(error_x + error_y, 0.5), columns=["error_xy"])

    pd_plot = commons.pandas_subplots(nrows=2, ncols=3, figsize=(25, 12), facecolor='w', fontsize=10)
    fig = pd_plot([pos_x, pos_y, yaw_theta, error_xy, particle_weight], ["position_x", "position_y", "yaw_rate", "cumulative_error", "particle_weight"])

    # output_pd.plot(figsize=(20, 10))
    path_ = os.path.join(save_dir, f"{sensor_type}.png")
    fig.savefig(path_)


if __name__ == "__main__":
    plot_stats(gt_prediction_data, sensor_type="gt_prediction_plot", save_dir="./images")
