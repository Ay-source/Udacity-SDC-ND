import glob
import cv2
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path_lsts = ["./object-dataset/", "./object-detection-crowdai/"]
file_exts = ["*.jpg", "*.csv"]
dst_path = "./data/extra/"
dst_dirs = ["vehicles/", "non-vehicles/"]
dst = ["extra1/", "extra2/"]
vehicle_section = ["car", "truck", "bike"]
non_vehicle_secion = ["person", "pedestrian", "trafficlight", "non-vehicle"]
resize_value = 64 # Size of window to extract
pick_size = [24, 128] # Size of non vehicle image to pick


def load_images(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def pick_non_vehicle_image(image, im_shape, resize_value, filtered_dfs):
    xmin = filtered_dfs["xmin"].tolist()
    xmax = filtered_dfs["xmax"].tolist()
    ymin = filtered_dfs["ymin"].tolist()
    ymax = filtered_dfs["ymax"].tolist()
    #print(len(xmin))
    x_range = []
    y_range = []
    for pos in range(len(xmin)):
        x_range = x_range + list(range(xmin[pos]-pick_size[1], xmax[pos]+pick_size[1]))
        y_range = y_range + list(range(ymin[pos]-pick_size[1], ymax[pos]+pick_size[1]))
    x_values = set(range(0, im_shape[1])) - set(x_range)
    y_values = set(range(0, im_shape[0])) - set(y_range)
    pick_range = list(range(pick_size[0], pick_size[1]))
    times = 0
    x_picked = [x_range[0], x_range[1]]
    y_picked = [y_range[0], y_range[1]]
    picked = False
    while any(x in x_range for x in x_picked) or any(x in y_range for x in y_picked):
        try:
            x_pick_value = random.choice(tuple(x_values))
            xdiff = random.choice(pick_range)
            x = [x_pick_value, x_pick_value+xdiff]
            x_picked = [range(x[0], x[1])]
            y_pick_value = random.choice(tuple(y_values))
            ydiff = random.choice(pick_range)
            y = [y_pick_value, y_pick_value+ydiff]
            y_picked = [range(y[0], y[1])]
        except IndexError:
            times = 21
        if times >= 20:
            return None#, "non-vehicle"
        times += 1
    non_vehicle_image = image[y[0]:y[1], x[0]:x[1], :]
    non_vehicle_image = cv2.resize(non_vehicle_image, (resize_value, resize_value))
    #plt.imshow(non_vehicle_image)
    #plt.show()
    return non_vehicle_image#, "non-vehicle"

def extract_labelled_image(images, df, resize_value):
    vehicles = []
    non_vehicles = []
    p = 0
    for image_path in tqdm(images):
        filtered_dfs = df[df["Frame"] == image_path.split("/")[-1]]
        image = load_images(image_path)
        im_shape = image.shape
        vehicle_data = []
        non_vehicle_data = []
        for filter_df in filtered_dfs.itertuples():
            if (filter_df.xmax <= filter_df.xmin) or (filter_df.ymax <= filter_df.ymin):
                continue
            if filter_df.Label.lower() in vehicle_section:
                vehicle_data.append(labelled_data(image, filter_df, resize_value))
            else:
                non_vehicle_data.append(labelled_data(image, filter_df, resize_value))
        while len(vehicle_data) > len(non_vehicle_data):
            non_data = pick_non_vehicle_image(image, im_shape, resize_value, filtered_dfs)
            if non_data is not None:
                non_vehicle_data.append(non_data)
            else:
                break
        #print(len(vehicle_data))
        #print(len(non_vehicle_data))
        vehicles = vehicles + vehicle_data
        non_vehicles = non_vehicles + non_vehicle_data
    return vehicles, non_vehicles
        
        

def labelled_data(image, boundary, resize_value):
    labelled_image = image[boundary.ymin:boundary.ymax, boundary.xmin:boundary.xmax, :]
    labelled_image = cv2.resize(labelled_image, (resize_value, resize_value))
    #plt.imshow(labelled_image)
    #plt.show()
    return labelled_image#, boundary.Label

def preprocess_csv(csv_data, crowdai):
    """
    return:
        A list data of dictionaries: Frame, xmin, ymin, xmax, ymax, Label
    """
    print("Processing csv data file")
    organized_data = []
    for row in csv_data.itertuples():
        if not crowdai:
            lst = row[1].split(" ")
            xmin, xmax, ymin, ymax, Label, Frame = int(lst[1]), int(lst[3]), int(lst[2]), int(lst[4]), lst[6][1:-1], lst[0]
            organized_data.append(
                {
                    "Frame": Frame,
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "Label": Label
                }
            )
        else:
            organized_data = csv_data
    print("Processing csv completed")
    if crowdai:
        return csv_data
    return pd.DataFrame(organized_data)

def save_data(images, loc):
    try:
        vehicle_images, non_vehicle_images = images[0], images[1]
        i = 1
        save_loc = dst_path + dst_dirs[0] + loc
        for image in tqdm(vehicle_images):
            file_name = save_loc+str(i)+".jpg"
            cv2.imwrite(file_name, image)
            i += 1
        i = 1
        save_loc = dst_path + dst_dirs[1] + loc
        for image in tqdm(non_vehicle_images):
            file_name = save_loc+str(i)+".jpg"
            cv2.imwrite(file_name, image)
            i += 1
        return True
    except Exception as e:
        print("Data wasn't saved completely")
        return False

    #return organized_data
def main():
    try:
        i = 0
        for path_lst in path_lsts:
            paths = glob.glob(path_lst+file_exts[0])
            csv_paths = glob.glob(path_lst+file_exts[1])
            #if i == 0:
            #    print("Skipping")
            #    i += 1
            #    continue
            if not i:
                df = pd.read_csv(csv_paths[-1], header=None)
            else:
                df = pd.read_csv(csv_paths[-1])
            df = preprocess_csv(df, i)
            print(df.head())
            print(f"Extracting path information for {path_lst[2:-2]}")
            vehicle_data, non_vehicle_data = extract_labelled_image(paths, df, resize_value)
            save_data((vehicle_data, non_vehicle_data), dst[i])
            i += 1
    except Exception as e:
        raise e
        print(f"Data doesn't exist Error message {e}")
    return vehicle_data, non_vehicle_data

if __name__ == "__main__":
    main()