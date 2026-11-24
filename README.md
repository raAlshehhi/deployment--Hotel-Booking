# StayWise: Streamlit deployment
This branch contains the application and dashboard CSV.
The app downloads the original trained model from its pinned version on main,
verifies its SHA-256 and size, and loads it with joblib memory mapping.
Training code and the saved model remain on the main branch.
No model retraining or prediction rule changes are included.
