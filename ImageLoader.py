import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def ImageLoader(folder, num, viewSide, included):
    IMG_SIZE = 64
    images = []
    view = []
    labels = []
    label_num = 0
    for sub_folder in os.listdir(folder):
        if sub_folder != ".DS_Store":
            for sub_sub_folder in os.listdir(f"{folder}" + "\\" + f"{sub_folder}"):
                if int(sub_sub_folder) == viewSide and (
                    (sub_folder == included[0])
                    or (sub_folder == included[1])
                    or (sub_folder == included[2])
                    or (sub_folder == included[3])
                ):
                    count = 0
                    for filename in os.listdir(
                        f"{folder}"
                        + "\\"
                        + f"{sub_folder}"
                        + "\\"
                        + f"{sub_sub_folder}"
                    ):
                        if filename != ".DS_Store":
                            count += 1
                            if (count >= num + 1) and (num != -1):
                                break

                            print(
                                f"{sub_folder} : {sub_sub_folder}: {filename}, {count}/{num}",
                                end="\r",
                                flush=True,
                            )
                            try:
                                img = (
                                    Image.open(
                                        f"{folder}"
                                        + "\\"
                                        + f"{sub_folder}"
                                        + "\\"
                                        + f"{sub_sub_folder}"
                                        + "\\"
                                        + f"{filename}"
                                    )
                                    .convert("L")
                                    .resize((IMG_SIZE, IMG_SIZE))
                                )
                                if False:

                                    if sub_folder == "No Tumor":
                                        labels.append(1)
                                    elif (
                                        sub_folder == "Pituitary"
                                        or sub_folder == "Meningioma"
                                        or sub_folder == "Glioma"
                                    ):
                                        labels.append(0)
                                else:
                                    labels.append(label_num)

                                images.append(np.array(img) / 255.0)
                                # labels.append(label_num)
                                view.append(int(sub_sub_folder))
                            except:
                                print(f"Error loading image: {folder} , {filename}")
                    label_num += 1
            print("")
    ints = np.arange(len(images))
    np.random.shuffle(ints)
    temp_images = []
    temp_labes = []
    for i in range(len(images)):
        temp_images.append(images[ints[i]])
        temp_labes.append(labels[ints[i]])
    images = temp_images
    labels = temp_labes

    return np.stack(images), np.array(labels)
