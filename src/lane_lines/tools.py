import numpy as np
from shapely import geometry


def get_slope_for_each_line(list_of_lines):
    """
        list_of_lines: [num_lines, (x1, y1, x2, y2)]
    """
    list_of_slopes = []
    for line in list_of_lines:
        x1, y1, x2, y2 = line
        list_of_slopes.append((y1 - y2) / (x1 - x2))
    return np.array(list_of_slopes).astype(np.float64)


def get_neg_pos_lines_and_their_respective_slopes(hough_lines_, slopes, img_shape, debug=False):
    
    """
    :param hough_lines_:
    :param slopes:
    :return:
    """
    assert(len(slopes) == len(hough_lines_))
    
    _, x_len = img_shape
    neg_idx = np.where(slopes < 0)
    pos_idx = np.where(slopes > 0)
    lines_with_neg_slope = hough_lines_[neg_idx]
    lines_with_pos_slope = hough_lines_[pos_idx]
    neg_slope_vals = slopes[neg_idx]
    pos_slope_vals = slopes[pos_idx]
    
    valid_index_neg_pos = np.unique(np.where(lines_with_neg_slope[:, 2] < (x_len / 2))[0])
    lines_with_neg_slope = lines_with_neg_slope[valid_index_neg_pos, :]
    neg_slope_vals = neg_slope_vals[valid_index_neg_pos]

    # Remove any line with positive slope occuring in the right lane
    valid_index_pos_pos = np.unique(np.where(lines_with_pos_slope[:, 0] > (x_len / 2))[0])
    lines_with_pos_slope = lines_with_pos_slope[valid_index_pos_pos, :]
    pos_slope_vals = pos_slope_vals[valid_index_pos_pos]
    
    if debug:
        print(f"\nlines_with_neg_slope:\n\t{lines_with_neg_slope}\n neg_slopes: \n\t{neg_slope_vals}", )
        print('')
        print(f"\nlines_with_pos_slope:\n\t{lines_with_pos_slope}\n pos_slopes: \n\t{pos_slope_vals}", )
        
    assert (len(lines_with_neg_slope) == len(neg_slope_vals))
    assert (len(lines_with_pos_slope) == len(pos_slope_vals))
    return lines_with_neg_slope, lines_with_pos_slope, neg_slope_vals, pos_slope_vals


def get_neg_pos_slope(neg_slopes, pos_slopes, aggregation_method=np.median, debug=False):
    negative_slope_val = aggregation_method(neg_slopes)
    positive_slope_val = aggregation_method(pos_slopes)
    if debug:
        print(
                f'\nnegative_slope_val = {negative_slope_val}, '
                f'\npositive_slope_val = {positive_slope_val} '
        )
    return negative_slope_val, positive_slope_val


def get_filtered_lines_with_neg_pos_slope(
        hough_lines_,
        slopes,
        min_slope_thresh,
        max_slope_thresh,
        slope_alignment,
        debug=False
):
    """
    Note: This technique works with no Regression, this technique just generates lines based on the obtained slope and
            qualified lines
            
    :param hough_lines_:                 [num_lines, (x1, y1, x2, y2)]
    :param slopes:                      [num_lines]
    :param min_slope_thresh:         float
    :param max_slope_thresh:         float
    :param slope_allignment:            str ('negative' or 'positive')
    :return:
    """
    if debug:
        print(f'len(hough_lines_)={len(hough_lines_)}, len(slopes)={len(slopes)}')
    assert (len(slopes) == len(hough_lines_))
    
    slope_range = (min_slope_thresh, max_slope_thresh)

    # Find only lines between the slope range
    if slope_alignment == 'neg':
        slope_buffer_idx = np.where((slopes < slope_range[0]) & (slopes > slope_range[1]))
    elif slope_alignment == "pos":
        slope_buffer_idx = np.where((slopes > slope_range[0]) & (slopes < slope_range[1]))
    else:
        raise ValueError('Only "neg" and "pos" allowed for slope alignment')
    
    lines_with_slope_buffer = hough_lines_[slope_buffer_idx]
    slope_with_buffer = slopes[slope_buffer_idx]
    
    if debug:
        print(f'\nslope_range = {slope_range}'
              f'\nlines_with_slope_buffer = {lines_with_slope_buffer}'
              f'\nslope_with_buffer = {slope_with_buffer}')
    
    return lines_with_slope_buffer, slope_with_buffer


def aggregate_lines_and_slopes(lines, slopes, aggregation_methods=np.median):
    """
    :param lines:                    [num_lines, (x1, x2, y1, y2)]
    :param slopes:                   [num_lines]
    :param aggregation_methods:      np.median or other method applied on vectors (np.median would get rid of outliers)
    :return:
    """
    return (
        np.expand_dims(aggregation_methods(lines, axis=0), axis=0).astype(np.int32),
        np.array([aggregation_methods(slopes)])
    )


def linear_extrapolation(x1, y1, x2, y2, ymin, ymax, slope=None, space_between_points=1, return_axis='xy'):
    """
    Note: Here we generate points for each line for the given slope
    TODO: This process has a problem though: since we use a single slope but varying bias for each points,
    the simple extrapolation model would also draw lines through the outlier points which may not exactly be on the line
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param ymin:
    :param ymax:
    :param slope:
    :param space_between_points:
    :param return_axis:
    :return:
    """
    slope = (y2 - y1) / (x2 - x1) if slope is None else slope
    # Calculate bias using one of the x and y
    bias = y1 - (slope * x1)
    
    points_in_line = abs(y2 - y1) // space_between_points
    
    # Create a range of y point between the two exterior points
    ys = np.linspace(min(y1, y2, ymin), max(y1, y2, ymax), num=points_in_line)
    
    # Calculate x using slope, bias and all y value
    xs = []
    for y in ys:
        xs.append((y - bias) / slope)
    
    if return_axis == 'xy':
        return np.column_stack((xs, ys)).astype(np.int32)
    else:
        return np.column_stack((ys, xs)).astype(np.int32)


def perform_extrapolation_for_given_slope(
        ymin, ymax,
        lines_to_extrapolate,
        slopes_to_extrapolate_with,
        extrapolation_method=linear_extrapolation,
        debug=False
):
    """
    :param ymin:                        (int) extrapolate from ymin
    :param ymax:                        (int) extrapolate to ymax
    :param lines_to_extrapolate:        [num_lines, (x1, y1, x2, y2)]
    :param slope_vals:                  [num_lines]
    :param extrapolation_method:        __func__ which by default uses the linear extrapolation
    
    :return:
        extrapolated_points         [num_points, (x1, y1)]
        extrapolated_polygons       [num_points, (x1, y1)]      ordered to form a polygon
    """
    assert(len(lines_to_extrapolate) == len(slopes_to_extrapolate_with)), (
        f'len(lines_to_extrapolate):{len(lines_to_extrapolate)} '
        f'doest equal len(slopes_to_extrapolate_with)={len(slopes_to_extrapolate_with)}'
    )
    extrapolated_points = np.concatenate([
        extrapolation_method(
                x1, y1, x2, y2, ymin=ymin, ymax=ymax, slope=slope_val, space_between_points=1, return_axis='xy'
        ) for (x1, y1, x2, y2), slope_val in zip(lines_to_extrapolate, slopes_to_extrapolate_with)
    ])

    if debug:
        print('\n[Shape] extrapolated_points: \n', extrapolated_points.shape)
    # poly_ = geometry.Polygon([[x, y] for x, y in extrapolated_points])
    # x, y = poly_.exterior.coords.xy
    # extrapolated_polygons = np.column_stack((x, y)).astype(np.int32)
    extrapolated_line = np.expand_dims(
            np.array([list(extrapolated_points[0]) + list(extrapolated_points[-1])]), axis=0
    )
    return extrapolated_points, extrapolated_line


def get_xy_points_from_lines(lines_nd):
    """
    list_of_lines = np.array([[x1, y1, x2, y2],[x1, y1, x2, y2]])
    """
    xpoints = lines_nd[:, 0::2].flatten()
    ypoints = lines_nd[:, 1::2].flatten()
    return xpoints, ypoints


def perform_extrapolation_with_polyfit(lines_nd, xmin, xmax, debug=False):
    """
    :param lines_nd: [[x1, y1, x2, y2], [x1, y1, x2, y2]]
    :return:
    """
    x_points, y_points = get_xy_points_from_lines(lines_nd)
    slope_bias = np.polyfit(x_points, y_points, deg=1)
    
    func_extrapolate = np.poly1d(slope_bias)
    
    # np.poly1d makes as equation of y=mx+b given m and b value from np.polyfit
    # So now we generate x values and use poly1d to generate respective y values
    x_new_neg = np.linspace(xmin, xmax, xmax - xmin)
    y_new_neg = func_extrapolate(x_new_neg)
    
    extrapolated_points = np.column_stack((x_new_neg, y_new_neg)).astype(np.int32)
    extrapolated_line = np.array([list(extrapolated_points[0]) + list(extrapolated_points[-1])])
    
    if debug:
        print(f'[Extrapolate with polyfit] '
              f'\n\tinput lines shape = {lines_nd.shape}'
              f'\n\tslope_bias = {slope_bias}'
              f'\n\txmin = {xmin} xmax = {xmax}'
              f'\n\textrapolated line = {extrapolated_line}')
    
    return extrapolated_line

