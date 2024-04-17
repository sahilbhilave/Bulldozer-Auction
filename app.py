import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('models/best_model_gb.pkl', 'rb'))

# Read unique values from the file

# Create a mapping from unique values to their encoded values


# Define encoding functions for categorical features
def encode_fi_model_desc(fi_model_desc):
    with open('asc_proper_desc_file.txt', 'r') as file:
    	unique_values = file.read().splitlines()

    encoded_values = {value: index for index, value in enumerate(unique_values)}
    return encoded_values.get(fi_model_desc, 0)

def encode_hydraulics_flow(hydraulics_flow):
    # Encode Hydraulics_Flow using appropriate encoding technique
    # For simplicity, let's assume one-hot encoding for now
    encoded_values = {'Blank': 0, 'High Flow': 1, 'Standard': 3, 'Low Flow': 2}  # Update with actual encoding
    return encoded_values.get(hydraulics_flow, 0)

def encode_fi_model_descriptor(fi_model_descriptor):
    # Encode fiModelDescriptor using appropriate encoding technique
    # For simplicity, let's assume binary encoding for now
    encoded_values = {
    'Blank': 0, '1': 1, '2': 2, '2.00E+00': 3, '2N': 4, '3': 5, '3.00E+00': 6, '3C': 7, '3L': 8, '3NO': 9,
    '4WD': 10, '4x4x4': 11, '5': 12, '6': 13, '6K': 14, '7': 15, '7.00E+00': 16, '7A': 17, '8': 18, ' 14FT': 19,
    'LGP': 20, 'SUPER': 21, 'XLT': 22, 'XT': 23, 'ZX': 24, '(BLADE RUNNER)': 25, 'A': 26, 'AE0': 27,
    'AVANCE': 28, 'B': 29, 'BE': 30, 'C': 31, 'CK': 32, 'CR': 33, 'CRSB': 34, 'CUSTOM': 35, 'DA': 36,
    'DELUXE': 37, 'DHP': 38, 'DINGO': 39, 'DLL': 40, 'DT': 41, 'DW': 42, 'E': 43, 'ESL': 44, 'G': 45,
    'GALEO': 46, 'H': 47, 'H5': 48, 'HD': 49, 'HF': 50, 'High Lift': 51, 'HighLift': 52, 'HSD': 53,
    'HT': 54, 'II': 55, 'III': 56, 'IT': 57, 'IV': 58, 'K': 59, 'K3': 60, 'K5': 61, 'KA': 62, 'KBNA': 63,
    'L': 64, 'LC': 65, 'LC8': 66, 'LCH': 67, 'LCR': 68, 'LCRTS': 69, 'LE': 70, 'LK': 71, 'LL': 72, 'LM': 73,
    'LN': 74, 'LongReach': 75, 'LR': 76, 'LRC': 77, 'LRR': 78, 'LS': 79, 'LT': 80, 'LU': 81, 'LX': 82,
    'M': 83, 'MC': 84, 'ME': 85, 'MH': 86, 'N': 87, 'NLC': 88, 'NSUC': 89, 'P': 90, 'PLUS': 91, 'PRO': 92,
    'RR': 93, 'RTS': 94, 'S': 95, 'SA': 96, 'SB': 97, 'SE': 98, 'SERIES3': 99, 'SITEMASTER': 100, 'SL': 101,
    'SLGP': 102, 'SM': 103, 'SR': 104, 'SRDZ': 105, 'SRLC': 106, 'SSR': 107, 'SU': 108, 'SUPER K': 109,
    'T': 110, 'TC': 111, 'TK': 112, 'TLB': 113, 'TM': 114, 'TP': 115, 'TURBO': 116, 'U': 117, 'USLC': 118,
    'V': 119, 'VHP': 120, 'VHP/AWD': 121, 'WLT': 122, 'WT': 123, 'X': 124, 'XD': 125, 'XL': 126, 'XLVP': 127,
    'XP': 128, 'XR': 129, 'XTV': 130, 'XW': 131, 'Y': 132, 'Z': 133, 'ZTS': 134, 'ZX': 135
}

    return encoded_values.get(fi_model_descriptor, 0)  # Return 0 if not found

def encode_product_size(product_size):
    # Encode ProductSize using appropriate encoding technique
    # For simplicity, let's assume ordinal encoding for now
    encoded_values = {'Blank': 0, 'Compact': 1, 'Large': 2, 'Large/Medium': 3, 'Medium': 4, 'Mini': 5,'Small': 6}  # Update with actual encoding
    return encoded_values.get(product_size, 0)  # Return 0 if not found

def encode_grouser(product_size):
    # Encode ProductSize using appropriate encoding technique
    # For simplicity, let's assume ordinal encoding for now
    encoded_values = {'Blank': 0, 'None or Unspecified': 1, 'Yes': 2}  # Update with actual encoding
    return encoded_values.get(product_size, 0)


# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')



# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting input features from the form
        machine_id = int(request.form.get('machineID', 0))
        model_id = int(request.form.get('modelID', 0))
        machine_hours_current_meter = int(request.form.get('machineHoursCurrentMeter', 0))
        hydraulics_flow = request.form.get('hydraulicsFlow', 'Standard')
        fi_model_desc = request.form.get('fiModelDesc', '')
        fi_model_descriptor = request.form.get('fiModelDescriptor', '')
        product_size = request.form.get('productSize', '')
        grouser_tracks = request.form.get('grouserTracks', 'None or Unspecified')

        # Processing categorical features
        fi_model_desc_encoded = encode_fi_model_desc(fi_model_desc)
        hydraulics_flow_encoded = encode_hydraulics_flow(hydraulics_flow)
        fi_model_descriptor_encoded = encode_fi_model_descriptor(fi_model_descriptor)
        product_size_encoded = encode_product_size(product_size)
        grouser_tracks_encoded = encode_grouser(grouser_tracks)

        # Creating feature array for prediction
        features = np.array([[machine_id, model_id, machine_hours_current_meter,
                              fi_model_desc_encoded, grouser_tracks_encoded,
                              hydraulics_flow_encoded, fi_model_descriptor_encoded,
                              product_size_encoded]])

        # Making prediction
        prediction = model.predict(features)

        # Returning prediction to the template
        return render_template('index.html', prediction_text=f'Predicted Sale Price: {prediction[0]:.2f}')
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('index.html', prediction_text=error_message)
        
if __name__ == "__main__":
    app.run(debug=True)

