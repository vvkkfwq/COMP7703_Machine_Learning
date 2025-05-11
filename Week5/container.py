from sklearn.model_selection import train_test_split
import pandas as pd
import os

# TODO: Load dataset
w3classif = pd.read_csv("./w3classif.csv", header=None, names=["x", "y", "label"])


# TODO: create a function to create 10 different shuffled train and test set pairs from the original dataset
def create_train_test_data(test_size=0.3):
    # For storing data
    trains, tests = [], []
    output_dir = "./splits"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(10):
        # TODO: Shuffle the dataset
        shuffled = w3classif.sample(frac=1, random_state=42 + i).reset_index(drop=True)

        # TODO: Split the dataset
        train_data, test_data = train_test_split(
            shuffled, test_size=test_size, random_state=42 + i
        )

        train_file = os.path.join(output_dir, f"w3classif_train_data_{i+1}.csv")
        test_file = os.path.join(output_dir, f"w3classif_test_data_{i+1}.csv")
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

        # Store data
        trains.append(train_data)
        tests.append(test_data)

    return trains, tests


create_train_test_data()
