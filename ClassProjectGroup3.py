# course: cmps3500
# CLASS Project
# PYTHON IMPLEMENTATION: BASIC DATA ANALYSIS
# date: 05/07/2025
# Student 1: Isaac Pitts
# Student 2: Andy Ceballos
# Student 3: Adrian Corona
# Student 4: Joseph Hernandez
# description: Implementation of Neural Network

#Libaray Imports
import pandas as pd
import numpy as np
import regex
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import glob
import concurrent.futures
from datetime import datetime
import warnings

#things we need for warnings and so on
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
# Global Variables as these are share checks for debugging and ending things
training_data = None                  # Raw loaded CSV
cleaned_data = None                   # Cleaned training set
model = None                          # Neural network model
encoder = None                        # One-hot encoder for features
target_encoder = None                 # One-hot encoder for labels
scaler = None                         # Normalization scaler
X = Y = None                          # Inputs and labels
testing_data = None                   # Raw testing set
cleaned_testing_data = None           # Cleaned testing set
predictions_df = None                 # Output predictions


#Timestamp Helper
def timestamp(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def read_csv(file_path):                # Function for timing the read of csv
    return pd.read_csv(file_path)

#Option (1) - Load Training Data
def load_training_data():
    global training_data
    good_dir_check = 0
    good_csv_check = 0
    current_dir = os.getcwd()
    # will loop until the directory and csv file are valid
    while(good_dir_check == 0 or good_csv_check == 0):
        print("\n")
        print("Current Directory: " + str(current_dir))
        print("Enter another directory path or press Enter to continue with current")
        cd_input = input("Directory: ").strip()
        if(cd_input == ""):
            select_dir = current_dir
        else:
            select_dir = cd_input

        # check that inputted directory is real
        if not (os.path.isdir(select_dir)):
            print("Error - Directory does not exist")
            continue
        else:
            good_dir_check = 1

        # gets all the csv files in a directory 
        csv_files = glob.glob(os.path.join(select_dir, "*.csv"))
        if(len(csv_files) == 0):
            print("Error - No CSV files in this directory. Please try another directory")
            good_dir_check = 0
        else:
            good_csv_check = 1

    # loads all the csv and lets user select one
    csv_index = -1
    try:
        for index in range(0, len(csv_files)):
            print("(" + str(index) + ") " + os.path.basename(csv_files[index]))
        csv_in = input("File Number: ")

        # save index of csv file that is a number only
        if(csv_in.isdigit()):
            csv_index = int(csv_in)

    except Exception as e:
            print("Error selecting file:", e)
        

    if(csv_index in range(len(csv_files))):
        start_time = time.time()
        timestamp("Starting Script")
        try:
             # START OF ERROR HANDLING #

            # Checking for how long csv file takes to read 
            max_read_seconds = 20
            missing_row_data = False
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(read_csv, csv_files[int(csv_in)])
                try:
                    training_data = future.result(timeout=max_read_seconds)
                except concurrent.futures.TimeoutError:
                    print("CSV File has Taken longer than 20 secs to read, cancelled read\n"
                    "Please choose a different CSV File to load")
                    return None
                
            # Checking for if there are columns in CSV but no rows
            if len(training_data) == 0 and len(training_data.columns) > 0:
                print("CSV File has columns but no row data, cancelled read\n"
                "Please choose a different CSV File to load")
                return None

            # Check for if a column has a constant value for all rows
            for columns in training_data.columns:
                if training_data[columns].eq(training_data[columns].iloc[0]).all():
                    print(f"Column {columns} in CSV has a constant value for all rows, can lead to possible division"
                          "by 0 errors and also adversely affect training. Please choose a different CSV File to load")
                    return None
                    
            # Checking for rows with missing entries, will still let you load the CSV #
            for columns in training_data.columns:
                if training_data[columns].isnull().any():
                    missing_row_data = True
            if (missing_row_data == True):
                print("Warning - Some columns in CSV contain missing row data, CSV Still Loaded")
            
            # END OF ERROR HANDLING #
            timestamp("Loading training data set")
            timestamp(f"Total Columns Read: {len(training_data.columns)}")
            timestamp(f"Total Rows Read: {len(training_data)}")
            print(f"\nTime to load is: {round(time.time() - start_time, 2)} seconds")
        except Exception as e:
            print("Error loading file:", e)
    else:
        print("Error the input is not a number")

#Option (2) - Clean training data
def clean_data():
    global training_data, cleaned_data
    if training_data is None:
        print("Please load training data first (Option 1)")
        return

    start_time = time.time()
    timestamp("Performing Data Clean Up")

    df = training_data.copy()

    # Begin cleaning with the changing of strings and such
    # Error Handling - This Try and Except statement should pick up any formula errors during cleaning
    try: 
        df = df.set_index('DR_NO')
        df = df.rename_axis('DR_NO_INDEX')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df['Date Rptd'] = df['Date Rptd'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))   #formating the times
        df['DATE OCC'] = df['DATE OCC'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) 
        df['AREA NAME'] = df['AREA NAME'].astype('string')
        df['Crm Cd Desc'] = df['Crm Cd Desc'].astype('string')
        df['Mocodes'] = df['Mocodes'].astype('string')
        df['Vict Sex'] = df['Vict Sex'].astype('string')   #convers theres to strings
        df['Vict Descent'] = df['Vict Descent'].astype('string')
        df['Premis Desc'] = df['Premis Desc'].astype('string')
        df['Weapon Desc'] = df['Weapon Desc'].astype('string')
        df['Status'] = df['Status'].astype('string')
        df['Status Desc'] = df['Status Desc'].astype('string')

        mapping = {
                'AA': 'Arrest',
                'AO': 'No Arrest',
                'JO': 'No Arrest',
                'JA': 'Arrest',
                'CC': 'No Arrest'
                }
        df['Target'] = df['Status'].map(mapping)   # map status to Arrest/No Arrest
        df = df[df['Status'] != 'IC']              # drop uncertain status

        #fill missing monocodes and count how many were recorded per incident
        df['Mocodes'] = df['Mocodes'].fillna('Unknown')
        df['Num_Mocodes'] = df['Mocodes'].apply(lambda x: 0 if x == 'Unknown' else len(x.split()))

        #dropping columns that are desc but not used in training
        df = df.loc[:, ~df.columns.str.contains('AREA NAME')]  #these all drops
        df = df.loc[:, ~df.columns.str.contains('Part 1-2')]
        df = df.loc[:, ~df.columns.str.contains('Crm Cd Desc')]
        df = df.loc[:, ~df.columns.str.contains('Premis Desc')]
        df = df.loc[:, ~df.columns.str.contains('Weapon Desc')]
        df = df.loc[:, ~df.columns.str.contains('Status Desc')]
        df = df.loc[:, ~df.columns.str.contains('Date Rptd')]

        # Extract month from DATE OCC
        df['MONTH OCC'] = df['DATE OCC'].dt.month
        #df = df.loc[:, ~df.columns.str.contains('DATE OCC')]

        df['TIME OCC'] = df['TIME OCC'].astype('string')
        df['TIME OCC'] = df['TIME OCC'].str.zfill(4)
        df['TEMP'] = pd.to_datetime(df['TIME OCC'], format='%H%M')
        df['HOUR'] = df['TEMP'].dt.hour

        #this is the mapping of hours so it can help the time and machine
        def map_time_numeric(hour):
            if 0 <= hour < 6:
                return 0  # Early Morning
            elif 6 <= hour < 12:
                return 1  # Morning
            elif 12 <= hour < 18:
                return 2  # Afternoon
            elif 18 <= hour < 24:
                return 3  # Night

        df['Time_Bucket_Num'] = df['HOUR'].apply(map_time_numeric)

        #this is for the months to group the months in the different seasons
        def month_to_season_numeric(month):
            # Error Handling - Adding Try statement for errors during month conversion
            try:
                if month in [12, 1, 2]:
                    return 0
                elif month in [3, 4, 5]:
                    return 1
                elif month in [6, 7, 8]:
                    return 2
                else:
                    return 3
            except Exception as e:
                print("Error during conversion of months to seasons: ", e)

        df['Season_Num'] = df['MONTH OCC'].apply(month_to_season_numeric)
    
        df = df.loc[:, ~df.columns.str.contains('TIME OCC')]
        df = df.loc[:, ~df.columns.str.contains('TEMP')]
        

        df = df.drop_duplicates() #dropping


        # Filter out invalid or missing values or data cleaning
        df.loc[df['Weapon Used Cd'].isna(), 'Weapon Used Cd'] = 0   # fill missing weapon with 0
        df = df[(df['Vict Age'] != 0) & (df['Vict Age'].notna())]   # remove bad/missing ages
        df = df[(df['Vict Sex'] != 'X') & (df['Vict Sex'] != 'H') & (df['Vict Sex'].notna())]     # cleans vict sex
        df = df[(df['Vict Descent'] != '-') & (df['Vict Descent'].notna())] # clean descent
        df = df.dropna() # drop remaining missing rows
        df = df[df['Vict Age'] > 5]
        df = df[df['Vict Age'] < 90]

        crm_count = df['Crm Cd'].value_counts() # count crime types
        bad_crm = crm_count[crm_count >= 100].index  # keep frequent crimes
        df = df[df['Crm Cd'].isin(bad_crm)]   # filter crimes

        df = df[df['Status'] != 'CC']  # remove CC status

        df.to_csv('WhatGroup3cleanlookslike.csv', index=True)  # save cleaned
        cleaned_data = df
        timestamp(f"Total Rows after cleaning is: {len(df)}")
        print(f"Time to process is: {round(time.time() - start_time, 2)} seconds")
    except Exception as e:
        print("Cleaning failed:", e)


# Adds new informative features to help the model learn patterns for FE
def feature_engineer(df):
    # Error Handling for model learning features
    try:
        df['WEEKDAY'] = df['DATE OCC'].dt.weekday
        df['Is_Weekend'] = df['WEEKDAY'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

        df['Crime_Hour_Bucket'] = pd.cut(df['HOUR'], bins=[0, 6, 12, 18, 24],
                                        labels=['Night', 'Morning', 'Afternoon', 'Evening'], include_lowest=True)

        df['Vict_Profile'] = df['Vict Sex'] + "_" + df['Vict Descent']

        return df
    except Exception as e:
        print("Error during feature_engineer function: ", e)


# Option (3) - Train Neural Network on the cleaned
def train_neural_network():
    global cleaned_data, model, encoder, target_encoder, scaler, X, Y

    if cleaned_data is None:
        print("Please clean the training data first using Option (2).") #if wrong thing is entered
        return

    #imports for the code and reason for here is debugging
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    from sklearn.utils.class_weight import compute_class_weight
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    start_time = time.time()
    timestamp("Training Neural Network")

    # Error handling for potential error during training
    try:
        cleaned_data = feature_engineer(cleaned_data) #FE
        # Variables for everything like num and cat
        target = ['Status']
        numerical = ['AREA', 'Rpt Dist No', 'Crm Cd', 'Vict Age', 'Premis Cd', 'Weapon Used Cd', 'Time_Bucket_Num', 'Season_Num']
        categorical = ['Target','Is_Weekend', 'Crime_Hour_Bucket', 'Vict_Profile']

        # Encode features
        encoder = OneHotEncoder(handle_unknown='ignore')  # Set up encoder to ignore unseen categories
        encoder.fit(cleaned_data[categorical])   # Fit encoder to categorical features
        encoded_df = pd.DataFrame(encoder.transform(cleaned_data[categorical]).toarray(), columns=encoder.get_feature_names_out(categorical))  # Transform to one-hot encoded array and convert to DataFrame
        df = pd.concat([cleaned_data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)  # Combine encoded data with the original

        # Encode target
        target_encoder = OneHotEncoder(sparse_output=False)     # Encoder for labels (Status column)
        target_encoder.fit(df[['Status']])                    # Fit label encoder
        Y = target_encoder.transform(df[['Status']])        # Convert labels to one-hot format

        # Final input
        features_in_model = numerical + list(encoded_df.columns)
        X = df[features_in_model].values

        # Scale numerical
        scaler = MinMaxScaler()            # Normalizes features between 0 and 1
        df[numerical] = scaler.fit_transform(df[numerical])          # Apply scaling to numerical features
        X = df[features_in_model].values                   # Final input matrix (numerical + encoded features)
 
        # THE MAIN AREA FOR TRAINING
        model = Sequential()
        model.add(Dense(38, input_dim=X.shape[1], activation='relu')) # Input layer with 38 units and ReLU activation
        model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
        model.add(Dense(48, activation="relu"))   # Hidden layer with 48 units
        model.add(Dense(96, activation="relu"))  # Hidden layer with 96 units
        model.add(Dense(96, activation="relu"))   # Another 96-unit hidden layer
        model.add(Dense(48, activation="relu"))    # Hidden layer with 48 units
        model.add(Dense(4, activation="softmax"))  # Output layer with 4 units (one per class), softmax for multi-class classification

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Class weights
        classes = target_encoder.categories_[0]        # Get class labels
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=cleaned_data['Status'])   # Compute balanced class weights
        class_weights = dict(zip(range(len(classes)), weights))   # Map each class to its weight

        #early stop if the code gets to noisy
        early_stop = EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)
        
        # Train information
        history = model.fit(X, Y, epochs=30, batch_size=32, validation_split=0.2, class_weight=class_weights, callbacks=[early_stop], verbose=0) #stops all that text stuff with verbose)
        # Train the model with class weights and early stopping


        # Plotting
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        #stop if it takes to long for error handling
        timestamp("Training completed.")
        elapsed = round(time.time() - start_time, 2)
        if elapsed > 90:
            timestamp("Warning: Training took over 90 seconds")

        # Predict on training set to evaluate metrics
        preds = model.predict(X)           # Predict on training data
        y_pred = target_encoder.inverse_transform(preds)    # Convert predictions from one-hot to labels
        y_true = target_encoder.inverse_transform(Y)       # Convert true labels from one-hot to labels

        #all the info with acc and prec
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)

        print("\n    Train NN:\n    ********")
        timestamp(f"Model Accuracy: {acc:.4f}")
        timestamp(f"Model Precision: {prec:.4f}")
        timestamp(f"Model Recall: {rec:.4f}")
        timestamp(f"Model f1_score: {f1:.4f}")
        timestamp("Model Confusion Matrix:")
        class_labels = ['AA', 'AO', 'JA', 'JO']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix (Training Data)")
        plt.show()
    except Exception as e:
        print("Error during training of Neural Network: ", e)

# Option (4) - Load and clean the test dataset
def load_testing_data():
    global testing_data
    good_dir_check = 0
    good_csv_check = 0
    current_dir = os.getcwd()
    while(good_dir_check == 0 or good_csv_check == 0):
        print("\n")
        print("Current Directory: " + str(current_dir))
        print("Enter another directory path or press Enter to continue with current")
        cd_input = input("Directory: ").strip()
        if(cd_input == ""):
            select_dir = current_dir
        else:
            select_dir = cd_input

        # check that inputted dir is real
        if not (os.path.isdir(select_dir)):
            print("Error - Directory does not exist")
            continue
        else:
            good_dir_check = 1

        #csv loading
        csv_files = glob.glob(os.path.join(select_dir, "*.csv"))
        if(len(csv_files) == 0):
            print("Error - No CSV files in this directory. Please try another directory")
            good_dir_check = 0
        else:
            good_csv_check = 1

    # loads all the csv and lets user select one
    csv_index = -1
    try:
        for index in range(0, len(csv_files)):
            print("(" + str(index) + ") " + os.path.basename(csv_files[index]))
        csv_in = input("File Number: ")

        # save index of csv file that is a number only
        if(csv_in.isdigit()):
            csv_index = int(csv_in)
    except Exception as e:
            print("Error selecting file:", e)

    if(csv_index in range(len(csv_files))):
        start_time = time.time()
        timestamp("Starting Script")
        try:
            # START OF ERROR HANDLING #

            # Checking for how long csv file takes to read #
            max_read_seconds = 20
            missing_row_data = False
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(read_csv, csv_files[int(csv_in)])
                try:
                    testing_data = future.result(timeout=max_read_seconds)
                except concurrent.futures.TimeoutError:
                    print("CSV File has Taken longer than 20 secs to read, cancelled read\n"
                    "Please choose a different CSV File to load")
                    return None
                
            # Checking for if there are columns in CSV but no rows
            if len(testing_data) == 0 and len(testing_data.columns) > 0:
                print("CSV File has columns but no row data, cancelled read\n"
                "Please choose a different CSV File to load")
                return None

            # Check for if a column has a constant value for all rows
            for columns in testing_data.columns:
                if testing_data[columns].eq(testing_data[columns].iloc[0]).all():
                    print(f"Column {columns} in CSV has a constant value for all rows, can lead to possible division"
                          "by 0 errors and also adversely affect training. Please choose a different CSV File to load")
                    return None
                    
            # Checking for rows with missing entries, will still load the CSV File #
            for columns in testing_data.columns:
                if testing_data[columns].isnull().any():
                    missing_row_data = True
            if (missing_row_data == True):
                print("Warning - Some columns in CSV contain missing row data, CSV Still Loaded")
                            
            # END OF ERROR HANDLING #
            timestamp("Loading testing data set")
            timestamp(f"Total Columns Read: {len(testing_data.columns)}")
            timestamp(f"Total Rows Read: {len(testing_data)}")
            print(f"\nTime to load is: {round(time.time() - start_time, 2)} seconds")
            clean_testing_data()
        except Exception as e:
            print("Error loading file:", e)
    else:
        print("Error the input is not a number")


# Cleans testing data just like training data
def clean_testing_data():
    global testing_data, cleaned_testing_data
    if testing_data is None:
        print("Please load the testing data first using Option (4).")
        return

    start_time = time.time()
    timestamp("Performing Clean-Up on Testing Data")

    df = testing_data.copy()
    # Error Handling - This Try statement will pick up errors during testing data cleaning
    try:
        df = df.set_index('DR_NO')
        df = df.rename_axis('DR_NO_INDEX')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df['Date Rptd'] = df['Date Rptd'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))
        df['DATE OCC'] = df['DATE OCC'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        df['AREA NAME'] = df['AREA NAME'].astype('string')
        df['Crm Cd Desc'] = df['Crm Cd Desc'].astype('string')
        df['Mocodes'] = df['Mocodes'].astype('string')
        df['Vict Sex'] = df['Vict Sex'].astype('string')
        df['Vict Descent'] = df['Vict Descent'].astype('string')
        df['Premis Desc'] = df['Premis Desc'].astype('string')
        df['Weapon Desc'] = df['Weapon Desc'].astype('string')
        df['Status'] = df['Status'].astype('string')
        df['Status Desc'] = df['Status Desc'].astype('string')

        mapping = {
                #'IC': 'No Arrest',
                'AA': 'Arrest',
                'AO': 'No Arrest',
                'JO': 'No Arrest',
                'JA': 'Arrest',
                'CC': 'No Arrest'
                }
        df['Target'] = df['Status'].map(mapping)
        df = df[df['Status'] != 'IC']

        df['Mocodes'] = df['Mocodes'].fillna('Unknown')
        df['Num_Mocodes'] = df['Mocodes'].apply(lambda x: 0 if x == 'Unknown' else len(x.split()))

        df = df.loc[:, ~df.columns.str.contains('AREA NAME')] #dropping
        df = df.loc[:, ~df.columns.str.contains('Part 1-2')]
        df = df.loc[:, ~df.columns.str.contains('Crm Cd Desc')]
        df = df.loc[:, ~df.columns.str.contains('Premis Desc')]
        df = df.loc[:, ~df.columns.str.contains('Weapon Desc')]
        df = df.loc[:, ~df.columns.str.contains('Status Desc')]
        df = df.loc[:, ~df.columns.str.contains('Date Rptd')]

        df['MONTH OCC'] = df['DATE OCC'].dt.month
        #df = df.loc[:, ~df.columns.str.contains('DATE OCC')]

        df['TIME OCC'] = df['TIME OCC'].astype('string')
        df['TIME OCC'] = df['TIME OCC'].str.zfill(4)
        df['TEMP'] = pd.to_datetime(df['TIME OCC'], format='%H%M')
        df['HOUR'] = df['TEMP'].dt.hour

        def map_time_numeric(hour):
            if 0 <= hour < 6:
                return 0
            elif 6 <= hour < 12:
                return 1
            elif 12 <= hour < 18:
                return 2
            elif 18 <= hour < 24:
                return 3
        df['Time_Bucket_Num'] = df['HOUR'].apply(map_time_numeric)

        def month_to_season_numeric(month):
            if month in [12, 1, 2]:
                return 0
            elif month in [3, 4, 5]:
                return 1
            elif month in [6, 7, 8]:
                return 2
            else:
                return 3
        df['Season_Num'] = df['MONTH OCC'].apply(month_to_season_numeric)

        df = df.loc[:, ~df.columns.str.contains('TIME OCC')]
        df = df.loc[:, ~df.columns.str.contains('TEMP')]

        df = df.drop_duplicates()
        df.loc[df['Weapon Used Cd'].isna(), 'Weapon Used Cd'] = 0
        df = df[(df['Vict Age'] != 0) & (df['Vict Age'].notna())]
        df = df[(df['Vict Sex'] != 'X') & (df['Vict Sex'] != 'H') & (df['Vict Sex'].notna())]
        df = df[(df['Vict Descent'] != '-') & (df['Vict Descent'].notna())]
        df = df.dropna()
        df = df[df['Vict Age'] > 5]
        df = df[df['Vict Age'] < 90]

        crm_count = df['Crm Cd'].value_counts()
        bad_crm = crm_count[crm_count >= 100].index
        df = df[df['Crm Cd'].isin(bad_crm)]
        df = df[df['Status'] != 'CC']

        cleaned_testing_data = df
        timestamp(f"Total Rows after cleaning is: {len(df)}")
        print(f"Time to process is: {round(time.time() - start_time, 2)} seconds")

    except Exception as e:
        print("Error during cleaning of testing data:", e)

# Option (5) - Generate predictions using the trained model
def generate_predictions():
    global cleaned_testing_data, model, encoder, target_encoder, scaler, predictions_df

    if model is None or cleaned_testing_data is None:
        print("Make sure you trained the model (Option 3) and cleaned testing data (Option 4).")
        return

    from sklearn.preprocessing import OneHotEncoder

    print("\n    Generate Predictions:")
    print("    ********************")
    start_time = time.time()
    timestamp("Generating prediction using selected Neural Network")

    # Error handling during prediction generating
    try:
        numerical = ['AREA', 'Rpt Dist No', 'Crm Cd', 'Vict Age', 'Premis Cd', 'Weapon Used Cd', 'Time_Bucket_Num', 'Season_Num']
        categorical = ['Target', 'Is_Weekend', 'Crime_Hour_Bucket', 'Vict_Profile']

        # Encode test features
        cleaned_testing_data = feature_engineer(cleaned_testing_data)
        encoded_df2 = pd.DataFrame(
            encoder.transform(cleaned_testing_data[categorical]).toarray(),
            columns=encoder.get_feature_names_out(categorical)
        )
        df2 = pd.concat([cleaned_testing_data.reset_index(drop=True), encoded_df2.reset_index(drop=True)], axis=1)

        # Final features
        features_in_model = numerical + list(encoded_df2.columns)
        df2[numerical] = scaler.transform(df2[numerical])
        X_test = df2[features_in_model].values

        # Predict
        preds = model.predict(X_test)
        y_predicted = target_encoder.inverse_transform(preds)

        # Prepare final DataFrame
        final_output = pd.DataFrame({
            "DR_NO_INDEX": cleaned_testing_data.index,
            "Status": y_predicted.flatten()
        })

        predictions_df = final_output
        predictions_df.to_csv("predictionClassProject3.csv", index=False)

        timestamp(f"Size of training set {len(X)}")
        timestamp(f"Size of testing set {len(X_test)}")
        timestamp("Predictions generated (predictionClassProject3.csv have been generated)....") ##FIX THIS
        timestamp(f"Size of testing set {len(X_test)}")
        timestamp(f"Time to predict: {round(time.time() - start_time, 2)} seconds")
    except Exception as e:
        print("Error during generate_prediction function: ", e)


# Option (6) - Show accuracy, precision, recall, F1 score, and confusion matrix
def print_accuracy_report():
    from sklearn.metrics import accuracy_score, root_mean_squared_error
    global predictions_df, cleaned_testing_data, target_encoder

    if predictions_df is None or cleaned_testing_data is None:
        print("Make sure you generated predictions with Option (5).")
        return

    timestamp("Evaluating predictions")

    # Error handling for processing accuracy report
    try: 
        # Decode actual and predicted
        y_true_encoded = target_encoder.transform(cleaned_testing_data[['Status']])
        y_true = target_encoder.inverse_transform(y_true_encoded)
        y_pred = predictions_df['Status'].values.reshape(-1, 1)

        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        correct_predictions = int(acc * len(y_true))

        # RMSE using numeric label mapping
        label_map = {'AA': 0, 'AO': 1, 'JA': 2, 'JO': 3}
        y_true_nums = [label_map[val] for val in y_true.flatten()]
        y_pred_nums = [label_map[val[0]] for val in y_pred]
        rmse = root_mean_squared_error(y_true_nums, y_pred_nums)

        print("\n    Accuracy of prediction is:")
        print("    **************************")
        timestamp(f"{correct_predictions} of correct predicted observations.")
        timestamp(f"{acc * 100:.2f} % of correct predicted observations.")
        timestamp(f"Model RMSE: {rmse:.4f}")
    except Exception as e:
        print("Error during processing of Accuracy Report: ", e)


def show_menu():
    print("\nMenu:")
    print("(1) Load training data")
    print("(2) Process (Clean) data")
    print("(3) Train NN")
    print("(4) Load testing data")
    print("(5) Generate Predictions")
    print("(6) Print Accuracy (Actual Vs Predicted)")
    print("(7) Quit")

def main():
    while True:
        show_menu()
        choice = input("Select an option: ").strip()
        if choice == '1':
            load_training_data()
        elif choice == '2':
            clean_data()
        elif choice == '3':
            train_neural_network()
        elif choice == '4':
            load_testing_data()
        elif choice == '5':
            generate_predictions()
        elif choice == '6':
            print_accuracy_report()
        elif choice == '7':
            print("Exiting.")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
