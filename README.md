# StayWise — Hotel Booking Cancellation Prediction

**Course:** Data Science for Business Applications

StayWise is a machine learning project that estimates whether a hotel booking may be canceled. It combines data analysis, model development, and a Streamlit application to support hotel staff.

## Business problem

Booking cancellations make it harder for hotels to plan occupancy and revenue. Identifying bookings that may need attention can help staff prioritize confirmation calls and other follow-up activities.

The prediction supports staff judgment; it does not guarantee that a guest will cancel.

## Project objectives

- Explore hotel booking patterns and cancellation rates.
- Prepare booking data for machine learning.
- Compare classification models.
- Estimate cancellation probabilities for individual bookings.
- Present results through an interactive web application.

## Dataset

The project uses hotel booking data stored in `hotel_bookings.csv`.

The original dataset contained 119,390 records. An additional 130,610 synthetic records were generated to create a combined dataset of 250,000 records. Synthetic records are generated examples, not additional real customer bookings.

The target variable is `is_canceled`:

- `0`: the booking was not canceled.
- `1`: the booking was canceled.

The prediction pipeline uses 22 input features covering:

- Reservation details, including hotel, arrival date, lead time, and nights booked.
- Guest details, including adults, children, babies, and country.
- Booking channels, customer type, meal package, and agent.
- Booking history, deposit type, reserved room type, and average daily rate.

## Data preparation

The preprocessing pipeline prepares numerical and categorical inputs:

- Numerical missing values are filled using the median.
- Numerical features are standardized.
- Categorical missing values are filled using the most frequent category.
- Categorical features are transformed using one-hot encoding.
- Unknown categorical values are handled by the encoder.

Preprocessing and classification are saved together in a pipeline so the application uses the same transformations as training.

## Model development

The project compared:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest

Models were evaluated using accuracy, precision, recall, and F1-score.

Random Forest was selected based on the model comparison. The saved classifier uses 200 decision trees.

The recorded cross-validation results for Random Forest were approximately:

| Metric | Result |
|---|---:|
| Accuracy | 77.93% |
| Precision | 75.62% |
| Recall | 59.97% |
| F1-score | 66.89% |

These results describe the project evaluation. They are not a guarantee of performance on future hotel bookings.

## Application features

### Overview

Displays booking summaries and charts that help users explore the historical dataset.

### Predict a Booking

Hotel staff enter information in three sections:

1. Reservation details
2. Guest details
3. Booking history

The application displays:

- Estimated cancellation probability.
- The model’s predicted class.
- A follow-up suggestion.
- The submitted booking details.

### About the Model

Explains the model, input features, evaluation results, and limitations.

## Follow-up threshold

The threshold determines which bookings are flagged for staff follow-up.

For example, with a threshold of 50%:

- A booking with an estimated cancellation probability of 65% is flagged.
- A booking with an estimated cancellation probability of 30% is not flagged.

Lowering the threshold flags more bookings. Increasing it flags fewer bookings.

The threshold changes the follow-up suggestion. It does not retrain the model or change the class returned by the model’s prediction method.

## Project files

| File | Purpose |
|---|---|
| `deployment.ipynb` | Data preparation, analysis, model development, and evaluation |
| `application.py` | Streamlit application |
| `hotel_bookings.csv` | Booking dataset |
| `hotel_booking_model.pkl` | Saved preprocessing and prediction pipeline |
| `requirements.txt` | Required Python packages |
| `.gitattributes` | Git LFS configuration for the model |
| `.gitignore` | Files excluded from Git |
| `README.md` | Project documentation |

## Packages

The project uses:

- pandas and NumPy for data preparation.
- scikit-learn for preprocessing, training, and evaluation.
- joblib for saving and loading the pipeline.
- Plotly, Matplotlib, and Seaborn for visualization.
- Streamlit for the web application.

The saved model requires `scikit-learn==1.6.1`.

## Run locally

Download or clone the project. Git LFS is required to retrieve the model from the main branch.

Install the packages:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run application.py
```

Keep the model and CSV in the same folder as `application.py`.

## Streamlit deployment

Application link:

[Open StayWise](https://deployment--hotel-booking.streamlit.app/)

The repository has two branches:

- `main` contains the full project, notebook, dataset, and saved model.
- `streamlit-runtime` contains the deployment version of the application.

The deployment version downloads the same trained model after startup instead of including it in the initial repository checkout. It verifies the downloaded file’s size and SHA-256 checksum, then loads it using memory mapping.

The model is cached across application interactions. A fresh cloud machine may need to download it again.

Python 3.11 is the intended cloud deployment environment. Successful deployment and prediction should be verified before the application is used for assessment.

## Limitations

- Cancellation probabilities are estimates, not certain outcomes.
- The dataset includes synthetic records.
- Randomly splitting the combined real and synthetic dataset may produce optimistic evaluation results if related examples appear in both sets.
- Evaluation on a separate, untouched real dataset would provide stronger evidence of generalization.
- Future bookings may differ from the historical training data.
- The large model can create cloud download and memory constraints.
- Follow-up suggestions require staff judgment and should not automatically trigger penalties or cancellation of reservations.

## Conclusion

StayWise demonstrates how data science can support hotel booking management by combining data preparation, classification, evaluation, and an interactive application.
