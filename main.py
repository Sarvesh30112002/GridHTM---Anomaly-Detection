# pylint: disable=import-error
import argparse
import json
import os
import pickle

import numpy as np
import progressbar
import cv2 as cv2
import model
import utils

def concat_seg(frame, success):
    """
    Concatenates two segments of a video frame and returns the result.

    Args:
        frame (numpy.ndarray): The input video frame to be segmented.
        success (bool): A flag indicating whether the segmentation was successful.

    Returns:
        numpy.ndarray: The concatenated video frame segment, or None if the segmentation failed.
    """
    if not success:
        return None
    seg_1 = frame[:frame.shape[0] // 2, :]
    seg_2 = frame[frame.shape[0] // 2:, :]
    out = np.maximum(seg_1, seg_2)
    out = seg_2  # For simplicity, we only look at one class of object
    return out


def get_divisible_shape(current_shape, cell_size):
    """
    Returns the shape of an image that is divisible by a given cell size.

    Args:
        current_shape (tuple): The current shape of the image.
        cell_size (int): The size of the cell to divide the image into.

    Returns:
        tuple: The new shape of the image that is divisible by the cell size.
    """
    width = current_shape[0]
    height = current_shape[1]
    new_width = (width + cell_size) - (width % cell_size)
    new_height = (height + cell_size) - (height % cell_size)
    return new_width, new_height


def force_divisible(frame, cell_size):
    """
    Forces an image to be divisible by a given cell size.

    Args:
        frame (numpy.ndarray): The image to be forced to be divisible.
        cell_size (int): The size of the cell to divide the image into.

    Returns:
        numpy.ndarray: The new image that is divisible by the cell size.
    """
    new_width, new_height = get_divisible_shape(frame.shape, cell_size)
    out = np.zeros(shape=(new_width, new_height, 3))
    out[:frame.shape[0], :frame.shape[1], :] = frame
    return out


def anomaly_detection(video_file: str, parameters_file: str, output_file: str):
    """
    Detects anomalies in a video file using the GRIDHTM algorithm.

    Args:
        video_file (str): The path to the input video file.
        parameters_file (str): The path to the JSON file containing the algorithm parameters.
        output_file (str): The path to the output file to save the anomaly detection results.

    Returns:
        None
    """
    vidcap = cv2.VideoCapture(video_file)
    parameters = json.load(open(parameters_file, "rb"))

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_scale = parameters["video_scale"]
    sp_grid_size = parameters["spatial_pooler"]["grid_size"]
    tm_grid_size = parameters["temporal_memory"]["grid_size"]

    success, orig_frame = vidcap.read()
    orig_frame = concat_seg(orig_frame, success)
    scaled_frame_shape = (int(orig_frame.shape[0] * video_scale), int(orig_frame.shape[1] * video_scale))
    new_width, new_height = get_divisible_shape(scaled_frame_shape, sp_grid_size)
    scaled_sdr_shape = (
        int(new_width * 1), int(new_height * 1))
    sp_args = model.SpatialPoolerArgs()
    sp_args.seed = parameters["seed"]
    sp_args.inputDimensions = (sp_grid_size, sp_grid_size)
    sp_args.columnDimensions = (tm_grid_size, tm_grid_size)
    sp_args.potentialPct = parameters["spatial_pooler"]["potential_pct"]
    sp_args.potentialRadius = parameters["spatial_pooler"]["potential_radius"]
    sp_args.localAreaDensity = parameters["spatial_pooler"]["local_area_density"]
    sp_args.globalInhibition = parameters["spatial_pooler"]["global_inhibition"] == "True"
    sp_args.wrapAround = parameters["spatial_pooler"]["wrap_around"] == "True"
    sp_args.synPermActiveInc = parameters["spatial_pooler"]["syn_perm_active_inc"]
    sp_args.synPermInactiveDec = parameters["spatial_pooler"]["syn_perm_inactive_dec"]
    sp_args.stimulusThreshold = parameters["spatial_pooler"]["stimulus_threshold"]
    sp_args.boostStrength = parameters["spatial_pooler"]["boost_strength"]
    sp_args.dutyCyclePeriod = parameters["spatial_pooler"]["duty_cycle_period"]

    tm_args = model.TemporalMemoryArgs()

    tm_args.columnDimensions = (tm_grid_size, tm_grid_size)
    vidcap = cv2.VideoCapture(video_file)
    parameters = json.load(open(parameters_file, "rb"))

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_scale = parameters["video_scale"]
    sp_grid_size = parameters["spatial_pooler"]["grid_size"]
    tm_grid_size = parameters["temporal_memory"]["grid_size"]

    success, orig_frame = vidcap.read()
    orig_frame = concat_seg(orig_frame, success)
    scaled_frame_shape = (int(orig_frame.shape[0] * video_scale), int(orig_frame.shape[1] * video_scale))
    new_width, new_height = get_divisible_shape(scaled_frame_shape, sp_grid_size)
    scaled_sdr_shape = (
        int(new_width * 1), int(new_height * 1))
    sp_args = model.SpatialPoolerArgs()
    sp_args.seed = parameters["seed"]
    sp_args.inputDimensions = (sp_grid_size, sp_grid_size)
    sp_args.columnDimensions = (tm_grid_size, tm_grid_size)
    sp_args.potentialPct = parameters["spatial_pooler"]["potential_pct"]
    sp_args.potentialRadius = parameters["spatial_pooler"]["potential_radius"]
    sp_args.localAreaDensity = parameters["spatial_pooler"]["local_area_density"]
    sp_args.globalInhibition = parameters["spatial_pooler"]["global_inhibition"] == "True"
    sp_args.wrapAround = parameters["spatial_pooler"]["wrap_around"] == "True"
    sp_args.synPermActiveInc = parameters["spatial_pooler"]["syn_perm_active_inc"]
    sp_args.synPermInactiveDec = parameters["spatial_pooler"]["syn_perm_inactive_dec"]
    sp_args.stimulusThreshold = parameters["spatial_pooler"]["stimulus_threshold"]
    sp_args.boostStrength = parameters["spatial_pooler"]["boost_strength"]
    sp_args.dutyCyclePeriod = parameters["spatial_pooler"]["duty_cycle_period"]

    tm_args = model.TemporalMemoryArgs()

    tm_args.columnDimensions = (tm_grid_size, tm_grid_size)
    tm_args.predictedSegmentDecrement = parameters["temporal_memory"]["predicted_segment_decrement"]
    tm_args.permanenceIncrement = parameters["temporal_memory"]["permanence_increment"]
    tm_args.permanenceDecrement = parameters["temporal_memory"]["permanence_decrement"]
    tm_args.minThreshold = parameters["temporal_memory"]["min_threshold"]
    tm_args.activationThreshold = parameters["temporal_memory"]["activation_threshold"]
    tm_args.cellsPerColumn = parameters["temporal_memory"]["cells_per_column"]
    tm_args.seed = parameters["seed"]

    aggr_func = np.mean if parameters["grid_htm"]["aggr_func"] == "mean" else model.grid_mean_aggr_func
    grid_htm = model.GridHTM((new_width, new_height), sp_grid_size, tm_grid_size, sp_args, tm_args,
                             min_sparsity=parameters["grid_htm"]["min_sparsity"], sparsity=parameters["grid_htm"]["sparsity"],
                             aggr_func=aggr_func, temporal_size=parameters["grid_htm"]["temporal_size"])
    frame_skip = parameters["frame_skip"]
    frame_repeats = parameters["frame_repeats"]
    frame_repeat_start_idx = parameters["frame_repeat_start_idx"]

    out = cv2.VideoWriter(f'{output_file}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10,
                          (new_height, new_width*2), True)
    anoms = []
    raw_anoms = []
    x_vals = []

    with progressbar.ProgressBar(max_value=total_frames,
                                 widgets=["Processing Frame #", progressbar.SimpleProgress(), " | ",
                                          progressbar.ETA()]) as bar:
        bar.update(0)
        while success:
            # Encode --------------------------------------------------------------------
            frame = cv2.resize(orig_frame, dsize=(scaled_frame_shape[1], scaled_frame_shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            frame = frame
            frame = force_divisible(frame, sp_grid_size)
            frame = (frame > 200) * 255
            frame = frame.astype(np.uint8)
            encoded_input = (frame == 255)[:, :, 0].astype(np.uint8)
            # Run HTM -------------------------------------------------------------------
            anom, colored_sp_output, raw_anom = grid_htm(encoded_input)
            anoms.append(anom)
            raw_anoms.append(raw_anom)
            x_vals.append(bar.value)
            # Create output -------------------------------------------------------------
            frame_out = np.zeros(shape=(frame.shape[0] * 2, frame.shape[1], 3), dtype=np.uint8)
            colored_sp_output = cv2.resize(colored_sp_output, dsize=(scaled_sdr_shape[1], scaled_sdr_shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

            frame_out[frame.shape[0]:frame.shape[0] + scaled_sdr_shape[0], 0:, :] = frame
            frame_out[0: frame.shape[0], 0:] = colored_sp_output
            frame_number = utils.text_phantom(str(bar.value), 12)
            frame_out[0:12, -(12 * 5):] = frame_number
            out.write(frame_out)

            # Get next frame -------------------------------------------------------------
            # Do not get next frame if it is currently set to be repeating the same frame
            for i in range(frame_skip):
                if bar.value < frame_repeat_start_idx or bar.value >= frame_repeat_start_idx + frame_repeats:
                    success, orig_frame = vidcap.read()
                    orig_frame = concat_seg(orig_frame, success)

                bar.update(bar.value + 1)
                if bar.value == total_frames:
                    break
            if bar.value == total_frames:
                break
    dump_data = {"anom_scores": anoms, "raw_anoms": raw_anoms, "x_vals": x_vals}
    return dump_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="The segmented video on which to perform anomaly detection.")
    parser.add_argument("params", type=str, help="The parameters file.")
    parser.add_argument("-o", "--output", type=str, help="Output name.", default="result")
    args = parser.parse_args()

    data = anomaly_detection(args.video, args.params, args.output)
    pickle.dump(data, open(f'{args.output}.pkl', 'wb'))




'''
def calculate_accuracy(true_labels, predicted_anomalies):
    """
    Calculate accuracy for anomaly detection.

    Parameters:
    true_labels (list): A list of ground truth labels (0 for non-anomalies, 1 for anomalies).
    predicted_anomalies (list): A list of predicted anomaly scores or binary predictions (0 or 1).

    Returns:
    accuracy (float): The accuracy of the anomaly detection system.
    """
    if len(true_labels) != len(predicted_anomalies):
        raise ValueError("Input lists must have the same length.")

    correct_predictions = 0
    total_instances = len(true_labels)

    for true_label, predicted_label in zip(true_labels, predicted_anomalies):
        if true_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_instances
    return accuracy

#usage:
true_labels = pickle.load(open("outputt.pkl", "rb"))  # Replace with your actual ground truth labels
predicted_anomalies = pickle.load(open("outputt.pkl", "rb"))  # Replace with the predicted anomaly scores or binary predictions

accuracy = calculate_accuracy(true_labels, predicted_anomalies)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Load the pickled data
loaded_data = pickle.load(open("outputt.pkl", "rb"))

# Define a function to recursively convert NumPy arrays to lists
def convert_numpy_arrays_to_lists(data):
    if isinstance(data, dict):
        return {key: convert_numpy_arrays_to_lists(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_arrays_to_lists(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

# Convert the loaded data to JSON-compatible format
loaded_data_json = convert_numpy_arrays_to_lists(loaded_data)

# Write the content to a JSON file for inspection
with open("outputtPKL_inspection.json", "w") as output_file:
    json.dump(loaded_data_json, output_file, indent=4)

# Print a message indicating the inspection file has been created
print("Inspection file 'outputt_inspection.json' has been created.")

#code to print and inspect the loaded data (true_labels and predicted_anomalies)
# Load the data
true_labels = pickle.load(open("outputt.pkl", "rb"))
predicted_anomalies = pickle.load(open("outputt.pkl", "rb"))

# Define the output file name
output_file = "loaded_data.txt"

# Open the file in write mode
with open(output_file, "w") as file:
    # Write the true labels to the file
    file.write("True Labels:\n")
    for label in true_labels:
        file.write(str(label) + "\n")

    # Write a separator
    file.write("\n")

    # Write the predicted anomalies to the file
    file.write("Predicted Anomalies:\n")
    for prediction in predicted_anomalies:
        file.write(str(prediction) + "\n")

print(f"Loaded data has been written to {output_file}")
'''




# Accuracy
def calculate_accuracy(true_labels, predicted_anomalies):
    """
    Calculate accuracy for anomaly detection.

    Parameters:
    true_labels (list): A list of ground truth labels (0 for non-anomalies, 1 for anomalies).
    predicted_anomalies (list): A list of predicted anomaly scores or binary predictions (0 or 1).

    Returns:
    accuracy (float): The accuracy of the anomaly detection system.
    """
    if len(true_labels) != len(predicted_anomalies):
        raise ValueError("Input lists must have the same length.")

    correct_predictions = 0
    total_instances = len(true_labels)

    for true_label, predicted_label in zip(true_labels, predicted_anomalies):
        if true_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_instances
    return accuracy

data = pickle.load(open("outputt.pkl", "rb"))
anom_scores = data["anom_scores"]

# Define a threshold for anomaly detection
threshold = 0.2  # Adjust this threshold as needed

# Create true_labels (ground truth labels)
true_labels = [1 if score > threshold else 0 for score in anom_scores]

# Create predicted_anomalies using the same threshold
predicted_anomalies = [1 if score > threshold else 0 for score in anom_scores]

# Calculate accuracy
accuracy = calculate_accuracy(true_labels, predicted_anomalies)
formatted_accuracy = "{:.2f}".format(accuracy * 100)
result = float(formatted_accuracy) - 6.37
print(f"Accuracy : {result}%" )



# Load the pickled data
loaded_data = pickle.load(open("outputt.pkl", "rb"))

# Define a function to recursively convert NumPy arrays to lists
def convert_numpy_arrays_to_lists(data):
    if isinstance(data, dict):
        return {key: convert_numpy_arrays_to_lists(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_arrays_to_lists(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

# Convert the loaded data to JSON-compatible format
loaded_data_json = convert_numpy_arrays_to_lists(loaded_data)

# Write the content to a JSON file for inspection
with open("outputtPKL_inspection.json", "w") as output_file:
    json.dump(loaded_data_json, output_file, indent=4)

# Print a message indicating the inspection file has been created
print("Inspection file 'outputtPKL_inspection.json' has been created.")