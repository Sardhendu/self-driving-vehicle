
import os
from moviepy.editor import VideoFileClip

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
    image = undistort(
            img=image, camera_matrix=CameraParams.camera_matrix,
            distortion_coefficients=CameraParams.distortion_coefficients
    )
    
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


def final_plots(save_dir):
    fig = commons.graph_subplots(nrows=1, ncols=2, figsize=(50, 10))(
            [CurvatureParams.left_lane_curvature_radii, CurvatureParams.right_lane_curvature_radii],
            ["Left Lane (curvature radius)", "Right Lane (curvature radius)"]
    )
    commons.save_matplotlib(f'{save_dir}/radius_of_curvature.png', fig)
    
    fig = commons.graph_subplots(nrows=1, ncols=2, figsize=(15, 6))(
            [CurvatureParams.left_line_curr_poly_variance, CurvatureParams.right_line_curr_poly_variance],
            ["left_lane_variance_curvature_change", "right_lane_curvature_change"]
    )
    commons.save_matplotlib(f'{save_dir}/change_in_curvature.png', fig)


if __name__ == "__main__":
    setting = "final"
    video_name = "project_video"
    input_video_path = f'./data/{video_name}.mp4'
    output_video_path = f'./data/{video_name}'
    output_img_dir = f"./data/debug_images/{video_name}"
    test_img_dir = f"./data/test_images/"
    
    os.makedirs(output_video_path, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------------------------
    # Final Video Pipeline
    # -------------------------------------------------------------------------------------------
    if setting == "final":
        clip2 = VideoFileClip(input_video_path)#.subclip(0, 3)
        yellow_clip = clip2.fl_image(final_pipeline)
        yellow_clip.write_videofile(f'{output_video_path}/{setting}_out.mp4', audio=False)
        final_plots(output_video_path)
    
    # -------------------------------------------------------------------------------------------
    # Debug Video
    # -------------------------------------------------------------------------------------------
    if setting == "warped":
        clip2 = VideoFileClip(input_video_path)#.subclip(0, 2)
        yellow_clip = clip2.fl_image(warped_output_video_pipeline)
        yellow_clip.write_videofile(f'{output_video_path}/{setting}_out.mp4', audio=False)
        final_plots(output_video_path)
    
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