from src import commons
from src.lane_line_advance.calibrate_camera import undistort
from src.lane_line_advance.params import CurvatureParams, CameraParams
from src.lane_line_advance.pipeline import preprocessing_pipeline, postprocessing_pipeline, lane_curvature_pipeline


def debug_pipeline(input_image_path, output_img_dir):
    image = commons.read_image(input_image_path)
    image = undistort(
            img=image, camera_matrix=CameraParams.camera_matrix,
            distortion_coefficients=CameraParams.distortion_coefficients
    )
    preprocessed_bin_image = preprocessing_pipeline(image, threshold_index=1, save_dir=output_img_dir)
    
    # -------------------------------------------------------------------------------------
    # Create Lane Curvature with 2nd degree polynomial
    # -------------------------------------------------------------------------------------
    left_x_new, right_x_new, y_new = lane_curvature_pipeline(
            preprocessed_bin_image, save_dir=output_img_dir, mode="debug"
    )

    # -------------------------------------------------------------------------------------
    # Un-warp (Project curvature points into the original image space)
    # -------------------------------------------------------------------------------------
    postprocessing_pipeline(image, left_x_new, right_x_new, y_new, save_dir=output_img_dir, mode="debug")


def final_pipeline(image):
    print('\n\n#--------------------------------------\n# Frame Initiate\n#--------------------------------------')
    # image = undistort(
    #         img=image, camera_matrix=CameraParams.camera_matrix,
    #         distortion_coefficients=CameraParams.distortion_coefficients
    # )
    
    preprocessed_bin_image = preprocessing_pipeline(image, threshold_index=1, save_dir=None)

    # -------------------------------------------------------------------------------------
    # Create Lane Curvature with 2nd degree polynomial
    # -------------------------------------------------------------------------------------
    left_x_new, right_x_new, y_new = lane_curvature_pipeline(
            preprocessed_bin_image, save_dir=None, mode="final"
    )

    # print(f'\n[Histogram] left_lane_pos_yx = {len(left_lane_pos_yx)}, right_lane_pos_yx = {len(right_lane_pos_yx)}')

    # -------------------------------------------------------------------------------------
    # Un-warp (Project curvature points into the original image space)
    # -------------------------------------------------------------------------------------
    out_image = postprocessing_pipeline(image, left_x_new, right_x_new, y_new, save_dir=output_img_dir, mode="final")
    return out_image


def warped_output_video_pipeline(image):
    print('\n\n#--------------------------------------\n# Frame Initiate\n#--------------------------------------')

    image = undistort(
            img=image, camera_matrix=CameraParams.camera_matrix,
            distortion_coefficients=CameraParams.distortion_coefficients
    )
    
    preprocessed_bin_image = preprocessing_pipeline(image, threshold_index=1, save_dir=None)
    
    # -------------------------------------------------------------------------------------
    # Create Lane Curvature with 2nd degree polynomial
    # -------------------------------------------------------------------------------------
    output = lane_curvature_pipeline(preprocessed_bin_image, save_dir=None, mode="warped")
    return output


def plot_curvature_radius_dist(save_path):
    fig = commons.graph_subplots(nrows=1, ncols=3, figsize=(50, 10))(
            [CurvatureParams.left_lane_curvature_radii, CurvatureParams.right_lane_curvature_radii],
            ["Left Lane (curvature radius)", "Right Lane (curvature radius)"]
    )
    commons.save_matplotlib(save_path, fig)

    
from moviepy.editor import VideoFileClip
setting = "final"
video_name = "project_video"
input_video_path = f'./data/{video_name}.mp4'
output_video_path = f'./data/{video_name}_{setting}_out.mp4'
output_img_dir = f"./data/debug_images/{video_name}"
test_img_dir = f"./data/test_images/"

# -------------------------------------------------------------------------------------------
# Final Video Pipeline
# -------------------------------------------------------------------------------------------
if setting == "final":
    clip2 = VideoFileClip(input_video_path)#.subclip(0, 10)
    yellow_clip = clip2.fl_image(final_pipeline)
    yellow_clip.write_videofile(output_video_path, audio=False)
    plot_curvature_radius_dist(f"./data/{video_name}_radius_curv.png")

# -------------------------------------------------------------------------------------------
# Debug Video
# -------------------------------------------------------------------------------------------
if setting == "warped":
    clip2 = VideoFileClip(input_video_path)#.subclip(0, 10)
    yellow_clip = clip2.fl_image(warped_output_video_pipeline)
    yellow_clip.write_videofile(output_video_path, audio=False)
    plot_curvature_radius_dist(f"./data/{video_name}_radius_curv.png")

# -------------------------------------------------------------------------------------------
# Debug Each Frame
# -------------------------------------------------------------------------------------------
# commons.fetch_image_from_video(
#         input_video_path, output_img_dir, time_list=[24.46]
# )

# time_list=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

if setting == "debug":
    test_image_name = "25.1"# "# "bridge_shadow" #"0"  #"0"

    # input_image_path = f'{test_img_dir}/{test_image_name}.jpg'
    input_image_path = f'{output_img_dir}/{test_image_name}.jpg'

    output_img_dir = f'{output_img_dir}/{test_image_name}'
    debug_pipeline(input_image_path, output_img_dir)



# TODO: When there are no pixels available in a window, increase the window size and recompute. Do this 3 times
#  increasing the aspect ration by 10% each time
# TODO: Find a way to start window search if poly line fit is not good. But how do you know that poly lines are not good
# TODO: WHen the r_and_s channel cannot find any points then process teh image through r_or_s