StayWise — Streamlit Application
Course: Data Science for Business Applications

StayWise helps hotel staff estimate the cancellation risk of a booking and identify bookings that may need follow-up.

Open the Streamlit application
About this branch
The streamlit-runtime branch contains the deployment version of the application. The complete project, training notebook, evaluation results, and saved model are on the main branch.

This branch uses the same trained model. It does not retrain the model or change its learned parameters.
Application pages
- Overview: explore historical booking summaries and charts.
- Predict a Booking: enter reservation details, guest details, and booking history to estimate cancellation risk.
- About the Model: read about the model, features, recorded evaluation results, and limitations.
Prediction results
After staff submit booking details, the application displays:
- Estimated cancellation probability.
- The model's predicted class.
- A follow-up suggestion based on the selected threshold.
- The submitted booking details.
For example, a booking with an estimated cancellation probability of 65% is flagged when the follow-up threshold is 50%.

Lowering the threshold flags more bookings. Raising it flags fewer bookings. The threshold changes the follow-up suggestion, not the model's predicted class.
Model
The application uses a saved Random Forest pipeline with 200 decision trees.

The pipeline includes missing-value imputation, numerical standardization, categorical encoding, and classification. It accepts 22 booking features.

For training methods and evaluation results, see the main project README.
Model download and loading
To keep the initial deployment checkout small, this branch downloads the original model after the application starts.

The application:
1. Reads the pinned model URL, expected size, and SHA-256 checksum from model_source.json.
2. Downloads the model to a temporary cache in small chunks, showing progress.
3. Checks the completed download against the expected size and checksum.
4. Loads the pipeline using joblib.load(..., mmap_mode="r").
5. Caches the loaded pipeline with st.cache_resource for reuse across interactions.
The first startup can take several minutes. A fresh cloud machine may need to download the model again. Downloading after startup avoids including the model in this branch's checkout; it does not remove cloud memory or download-bandwidth limits.
Deployment files
File	Purpose
application.py	Streamlit interface, model download, and prediction logic
hotel_bookings.csv	Historical booking data for the dashboard
model_source.json	Pinned model URL, size, and SHA-256 checksum
requirements.txt	Python dependencies
.gitignore	Excludes local environments, downloaded models, and temporary files
README.md	Deployment documentation


Streamlit Community Cloud settings
Setting	Value
Repository	raAlshehhi/deployment--Hotel-Booking
Branch	streamlit-runtime
Main file path	application.py
Python version	3.11
scikit-learn version	1.6.1


The saved model is downloaded from a pinned commit in the main project's history. Preserve that model and commit so the download remains available.

Verify a prediction on the public app before treating deployment as complete. Compare the result with the original local application using identical inputs.
Run this version locally
Clone this deployment branch:
git clone --branch streamlit-runtime --single-branch https://github.com/raAlshehhi/deployment--Hotel-Booking.git
cd deployment--Hotel-Booking
Using a Python 3.11 environment, install the packages and run the app:
python -m pip install -r requirements.txt
python -m streamlit run application.py
An internet connection is required for the first model download unless the original model is already available beside application.py.
Limitations
- Predictions are estimates and should support staff judgment.
- The project dataset includes synthetic records; evaluation results may not represent future real bookings.
- Large-model downloads and memory use can affect startup and availability.
- Follow-up flags should not automatically trigger penalties or cancellation of reservations.
Updating this deployment
Commit application changes to streamlit-runtime to update this deployment. Changes made only on main do not automatically update the deployed code.

Keep the full project explanation and training materials on main.
