import pickle

# Load the model
filename = "model.h5"
with open(filename, "rb") as file:
    model = pickle.load(file)

# Save the model temporarily
with open("temp_model.pkl", "wb") as temp_file:
    pickle.dump(model, temp_file)