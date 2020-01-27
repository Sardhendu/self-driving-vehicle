import numpy as np
from src import commons
from src.lane_line_advance.lane_curvature import ModelParams
from src.lane_line_advance.pipeline import PreprocessingPipeline, PostprocessingPipeline
from src.lane_line_advance.lane_curvature import LaneCurvature, fetch_start_position_with_hist_dist


def debug_pipeline(input_image_path, output_img_dir):
    image = commons.read_image(input_image_path)
    preprocess_pipeline = PreprocessingPipeline(image, save_path=f'{output_img_dir}/preprocess.png')
    postprocess_pipeline = PostprocessingPipeline(image, save_path=f'{output_img_dir}/postprocess.png')
    
    preprocess_pipeline.warp()
    preprocessed_bin_image = preprocess_pipeline.preprocess()
    preprocessed_bin_image = preprocessed_bin_image.astype(np.int32)
    preprocess_pipeline.plot()
    
    # -------------------------------------------------------------------------------------
    # Get histogram distribution to determine start point for sliding window
    # -------------------------------------------------------------------------------------
    left_lane_pos_yx, right_lane_pos_yx = fetch_start_position_with_hist_dist(
            preprocessed_bin_image.copy(), save_path=f"{output_img_dir}/histogram_dist.png"
    )
    
    # -------------------------------------------------------------------------------------
    # Create Lane Curvature with 2nd degree polynomial
    # -------------------------------------------------------------------------------------
    lane_curvature = LaneCurvature(
            preprocessed_bin_image=preprocessed_bin_image,
            left_lane_pos_yx=left_lane_pos_yx,
            right_lane_pos_yx=right_lane_pos_yx,
            window_size=(70, 130),
            margin=100,
            save_dir=output_img_dir,
            pipeline="debug"
    )
    lane_curvature.find_lane_points()
    lane_curvature.fit()
    y_new, left_x_new, right_x_new = lane_curvature.predict()
    print(f'\n[Lane Curvature] '
          f'y_new = {len(y_new)}, left_x_new = {len(left_x_new)}, right_x_new = {len(right_x_new)}')
    lane_curvature.plot(y_new, left_x_new, y_new, right_x_new)
    lane_curvature.measure_radius_in_meter()
    # -------------------------------------------------------------------------------------
    # Un-warp (Project curvature points into the original image space)
    # -------------------------------------------------------------------------------------
    postprocess_pipeline.unwarp()
    print('[Mtrix] postprocess_pipeline: \n', postprocess_pipeline.M)
    left_lane_points, right_lane_points = postprocess_pipeline.transform_lane_points(
            left_lane_points=np.column_stack((left_x_new, y_new)),
            right_lane_points=np.column_stack((right_x_new, y_new))
    )
    
    print(
            f'\n[Lane Detection] '
            f'left_lane = {len(left_lane_points)}, right_lane = {len(right_lane_points)}'
    )
    _ = postprocess_pipeline.draw_lane_mask(
            left_lane_points, right_lane_points,
            ModelParams.left_lane_curvature_radii_curr,
            ModelParams.right_lane_curvature_radii_curr
    )
    postprocess_pipeline.plot()


def final_pipeline(image):
    print('\n\n#--------------------------------------\n# Frame Initiate\n#--------------------------------------')
    preprocess_pipeline = PreprocessingPipeline(image)
    postprocess_pipeline = PostprocessingPipeline(image)
    
    preprocess_pipeline.warp()
    preprocessed_bin_image = preprocess_pipeline.preprocess()
    preprocessed_bin_image = preprocessed_bin_image.astype(np.int32)
    
    # -------------------------------------------------------------------------------------
    # Get histogram distribution to determine start point for sliding window
    # -------------------------------------------------------------------------------------
    left_lane_pos_yx, right_lane_pos_yx = fetch_start_position_with_hist_dist(
            preprocessed_bin_image.copy(), save_path=None
    )
    # print(f'\n[Histogram] left_lane_pos_yx = {len(left_lane_pos_yx)}, right_lane_pos_yx = {len(right_lane_pos_yx)}')
    
    # -------------------------------------------------------------------------------------
    # Create Lane Curvature with 2nd degree polynomial
    # -------------------------------------------------------------------------------------
    lane_curvature = LaneCurvature(
            preprocessed_bin_image=preprocessed_bin_image,
            left_lane_pos_yx=left_lane_pos_yx,
            right_lane_pos_yx=right_lane_pos_yx,
            window_size=(70, 130),
            margin=100,
            save_dir=None,
            pipeline="final"
    )
    lane_curvature.find_lane_points()
    lane_curvature.fit()
    y_new, left_x_new, right_x_new = lane_curvature.predict()
    # print(f'\n[Lane Curvature] '
    #       f'y_new = {len(y_new)}, left_x_new = {len(left_x_new)}, right_x_new = {len(right_x_new)}')
    lane_curvature.measure_radius_in_meter()
    # -------------------------------------------------------------------------------------
    # Un-warp (Project curvature points into the original image space)
    # -------------------------------------------------------------------------------------
    postprocess_pipeline.unwarp()
    left_lane_points, right_lane_points = postprocess_pipeline.transform_lane_points(
            left_lane_points=np.column_stack((left_x_new, y_new)),
            right_lane_points=np.column_stack((right_x_new, y_new))
    )
    # print(
    #         f'\n[Lane Detection] '
    #         f'left_lane = {len(left_lane_points)}, right_lane = {len(right_lane_points)}'
    # )
    out_image = postprocess_pipeline.draw_lane_mask(
            left_lane_points, right_lane_points,
            ModelParams.left_lane_curvature_radii_curr,
            ModelParams.right_lane_curvature_radii_curr
    )
    return out_image


def warped_output_video_pipeline(image):
    print('\n\n#--------------------------------------\n# Frame Initiate\n#--------------------------------------')
    preprocess_pipeline = PreprocessingPipeline(image)
    # postprocess_pipeline = PostprocessingPipeline(image)
    
    preprocess_pipeline.warp()
    preprocessed_bin_image = preprocess_pipeline.preprocess()
    preprocessed_bin_image = preprocessed_bin_image.astype(np.int32)
    
    # -------------------------------------------------------------------------------------
    # Get histogram distribution to determine start point for sliding window
    # -------------------------------------------------------------------------------------
    left_lane_pos_yx, right_lane_pos_yx = fetch_start_position_with_hist_dist(
            preprocessed_bin_image.copy(), save_path=None
    )
    # print(f'\n[Histogram] left_lane_pos_yx = {len(left_lane_pos_yx)}, right_lane_pos_yx = {len(right_lane_pos_yx)}')
    
    # -------------------------------------------------------------------------------------
    # Create Lane Curvature with 2nd degree polynomial
    # -------------------------------------------------------------------------------------
    lane_curvature = LaneCurvature(
            preprocessed_bin_image=preprocessed_bin_image,
            left_lane_pos_yx=left_lane_pos_yx,
            right_lane_pos_yx=right_lane_pos_yx,
            window_size=(70, 130),
            margin=100,
            save_dir=None,
            pipeline="preprocess"
    )
    lane_curvature.find_lane_points()
    lane_curvature.fit()
    y_new, left_x_new, right_x_new = lane_curvature.predict()
    lane_curvature.plot(y_new, left_x_new, y_new, right_x_new)
    # print(f'\n[Lane Curvature] '
    #       f'y_new = {len(y_new)}, left_x_new = {len(left_x_new)}, right_x_new = {len(right_x_new)}')
    # lane_curvature.measure_radii_in_meter()
    return lane_curvature.preprocessed_img_plot.image



def plot_curvature_radius_dist(save_path):
    fig = commons.graph_subplots(nrows=1, ncols=3, figsize=(50, 10))(
            [ModelParams.left_lane_curvature_radii, ModelParams.right_lane_curvature_radii],
            ["Left Lane (curvature radius)", "Right Lane (curvature radius)"]
    )
    commons.save_matplotlib(save_path, fig)

    
from moviepy.editor import VideoFileClip
video_name = "challenge_video"
input_video_path = f'./data/{video_name}.mp4'
output_video_path = f'./data/{video_name}_warped_out.mp4'
output_img_dir = f"./data/debug_images/{video_name}"


# -------------------------------------------------------------------------------------------
# Debug Video
# -------------------------------------------------------------------------------------------
# clip2 = VideoFileClip(input_video_path).subclip(0, 10)
# yellow_clip = clip2.fl_image(warped_output_video_pipeline)
# yellow_clip.write_videofile(output_video_path, audio=False)
# plot_curvature_radius_dist(f"./data/{video_name}_radius_curv.png")

# -------------------------------------------------------------------------------------------
# Debug Each Frame
# -------------------------------------------------------------------------------------------
# commons.fetch_image_from_video(
#         input_video_path, output_img_dir, time_list=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# )

test_image_name = "0"
input_image_path = f'{output_img_dir}/{test_image_name}.jpg'
output_img_dir = f'{output_img_dir}/{test_image_name}'
debug_pipeline(input_image_path, output_img_dir)
