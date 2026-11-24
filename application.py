
import calendar
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Set the page title and layout.
st.set_page_config(
    page_title="StayWise | Hotel Booking Prediction",
    page_icon="🏨",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "hotel",
    "lead_time",
    "arrival_date_month",
    "arrival_date_year",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "customer_type",
    "adr"
]

MONTHS = list(calendar.month_name)[1:]


# Set the colors and appearance.
st.markdown(
    """
    <style>
        .stApp {
            background: #f5f7fb;
            color: #17233b;
        }

        .block-container {
            padding-top: 5rem;
            padding-bottom: 3rem;
            max-width: 1350px;
        }

        [data-testid="stSidebar"] {
            background: #101d32;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] .stMarkdown {
            color: #f3f6fc;
        }

        .hero {
            background: linear-gradient(
                120deg, #101d32, #234565
            );
            padding: 32px;
            border-radius: 18px;
            margin-bottom: 24px;
        }

        .hero-label {
            color: #83dfcf;
            font-size: 12px;
            letter-spacing: 3px;
            font-weight: 700;
        }

        .hero h1 {
            color: white;
            font-size: 36px;
            margin: 10px 0;
        }

        .hero p {
            color: #d0dbea;
            margin-bottom: 0;
        }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #e3e8f0;
            border-radius: 14px;
            padding: 18px;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: #17233b;
        }

        div[data-testid="stForm"] {
            background: white;
            border: 1px solid #e3e8f0;
            border-radius: 16px;
            padding: 24px;
        }

        .stButton button,
        .stFormSubmitButton button {
            border-radius: 10px;
            font-weight: 600;
        }

        .model-ready {
            background: #123c3d;
            color: #d9fff3;
            border-radius: 12px;
            padding: 15px;
        }

        .model-missing {
            background: #54411d;
            color: #fff1c9;
            border-radius: 12px;
            padding: 15px;
        }

        .footer {
            color: #728098;
            text-align: center;
            font-size: 12px;
            margin-top: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def show_heading(label, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-label">{label}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_chart(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#17233b"),
        margin=dict(l=15, r=15, t=40, b=15),
        legend_title_text=""
    )
    return fig


# Load the saved pipeline.
@st.cache_resource
def load_model(path, modified_time):
    print("Starting model loading", flush=True)

    limit_file = Path("/sys/fs/cgroup/memory.max")

    if limit_file.exists():
        limit = limit_file.read_text().strip()

        if limit != "max":
            print(
                f"Cloud memory limit: "
                f"{int(limit) / (1024 ** 3):.2f} GiB",
                flush=True
            )

    model = joblib.load(path, mmap_mode="r")

    print("Model loaded successfully", flush=True)
    return model

classifier = None
model_error = None

model_path = BASE_DIR / "hotel_booking_model.pkl"

if model_path.exists():
    try:
        classifier = load_model(
            str(model_path),
            model_path.stat().st_mtime_ns
        )
    except Exception as error:
        model_error = str(error)


# Load the existing CSV for the dashboard.
@st.cache_data
def load_history(path, modified_time):
    df = pd.read_csv(
        path,
        na_values=["null"],
        low_memory=False
    )

    if "hotel" in df.columns:
        df["hotel"] = (
            df["hotel"].astype("string").str.strip()
        )

    if "arrival_date_month" in df.columns:
        df["arrival_date_month"] = (
            df["arrival_date_month"]
            .astype("string")
            .str.strip()
            .str.title()
        )

    if "customer_type" in df.columns:
        df["customer_type"] = (
            df["customer_type"]
            .astype("string")
            .str.strip()
        )

    if "adr" in df.columns:
        df["adr"] = pd.to_numeric(
            df["adr"].astype("string")
                     .str.replace("EUR ", "", regex=False)
                     .str.strip(),
            errors="coerce"
        )

    if "is_canceled" in df.columns:
        df["is_canceled"] = pd.to_numeric(
            df["is_canceled"],
            errors="coerce"
        )

    return df


# Match the agent input to the saved pipeline.
def prepare_agent(value, model):
    text = value.strip()

    if text:
        try:
            number = float(text)
        except ValueError:
            raise ValueError(
                "Enter a valid agent number, or leave it blank."
            )

        if (
            not np.isfinite(number)
            or number < 0
            or number % 1 != 0
        ):
            raise ValueError(
                "The agent ID must be a nonnegative whole number."
            )
    else:
        number = np.nan

    # Check whether the pipeline encodes agent as a category.
    for step in getattr(model, "named_steps", {}).values():
        if not isinstance(step, ColumnTransformer):
            continue

        for _, transformer, columns in step.transformers_:
            if not isinstance(
                columns,
                (list, tuple, pd.Index, np.ndarray)
            ):
                continue

            if "agent" not in columns:
                continue

            position = list(columns).index("agent")

            parts = [transformer]
            parts += list(
                getattr(
                    transformer, "named_steps", {}
                ).values()
            )

            for part in parts:
                if not isinstance(part, OneHotEncoder):
                    continue

                categories = part.categories_[position]

                if not text:
                    for category in categories:
                        if str(category).lower() == "unknown":
                            return category

                    return np.nan

                for category in categories:
                    try:
                        if float(category) == number:
                            return category
                    except (TypeError, ValueError):
                        pass

                if (
                    len(categories) > 0
                    and isinstance(categories[0], str)
                ):
                    return str(int(number))

    return number


def check_model():
    if classifier is not None:
        return True

    if model_error:
        st.error(
            "The model could not be opened. "
            "Check that the app uses the same Python environment "
            "as your training notebook."
        )

        with st.expander("Show error details"):
            st.code(model_error)
    else:
        st.error(
            "Place hotel_booking_model.pkl "
            "in the same folder as this app."
        )

    return False


# Add the menu and follow-up setting.
with st.sidebar:
    st.markdown("## 🏨 StayWise")
    st.caption("HOTEL BOOKING PREDICTION")

    st.divider()

    page = st.radio(
        "Choose a page",
        [
            "Overview",
            "Predict a Booking",
            "About the Model"
        ]
    )

    st.divider()

    threshold = st.slider(
        "When should a booking need follow-up?",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        format="%d%%"
    )

    st.caption(
        f"Bookings with an estimated cancellation probability "
        f"of {threshold}% or more are flagged for follow-up."
    )

    st.caption(
        "Lower this value to flag more bookings. "
        "Raise it to flag fewer bookings."
    )

    st.divider()

    if classifier is not None:
        st.markdown(
            '<div class="model-ready">'
            '✓ Model ready for predictions'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="model-missing">'
            'Model is not ready'
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    show_heading(
    "BOOKING OVERVIEW",
    "Understand your hotel bookings.",
    "Explore booking patterns and cancellation rates."
)

    csv_path = None

    for filename in [
        "hotel_bookings.csv",
        "hotel_bookings_250000_messy.csv"
    ]:
        candidate = BASE_DIR / filename

        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        st.info(
            "To display the dashboard, place hotel_bookings.csv "
            "in the same folder as this app."
        )

        st.write(
            "You can still use Predict a Booking "
            "without the dashboard CSV."
        )

    else:
        try:
            df = load_history(
                str(csv_path),
                csv_path.stat().st_mtime_ns
            )

            required = [
                "hotel",
                "arrival_date_month",
                "customer_type",
                "adr",
                "is_canceled"
            ]

            missing = [
                column for column in required
                if column not in df.columns
            ]

            if missing:
                st.error(
                    "The dashboard CSV is missing: "
                    + ", ".join(missing)
                )

            else:
                first, second, third = st.columns(3)

                with first:
                    hotels = st.multiselect(
                        "Hotel",
                        sorted(
                            df["hotel"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                    )

                with second:
                    months = st.multiselect(
                        "Arrival month",
                        MONTHS
                    )

                with third:
                    customers = st.multiselect(
                        "Customer type",
                        sorted(
                            df["customer_type"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                    )

                filtered = df.copy()

                if hotels:
                    filtered = filtered[
                        filtered["hotel"].isin(hotels)
                    ]

                if months:
                    filtered = filtered[
                        filtered["arrival_date_month"].isin(
                            months
                        )
                    ]

                if customers:
                    filtered = filtered[
                        filtered["customer_type"].isin(
                            customers
                        )
                    ]

                if filtered.empty:
                    st.info(
                        "No bookings match your selected filters."
                    )

                else:
                    outcomes = filtered[
                        filtered["is_canceled"].isin([0, 1])
                    ].copy()

                    rates = filtered.loc[
                        filtered["adr"] >= 0,
                        "adr"
                    ]

                    first, second, third, fourth = st.columns(4)

                    first.metric(
                        "Total bookings",
                        f"{len(filtered):,}"
                    )

                    second.metric(
                        "Canceled bookings",
                        f"{int(outcomes['is_canceled'].sum()):,}"
                    )

                    rate = outcomes["is_canceled"].mean()

                    third.metric(
                        "Cancellation rate",
                        (
                            f"{rate:.1%}"
                            if pd.notna(rate)
                            else "Unavailable"
                        )
                    )

                    fourth.metric(
                        "Average nightly rate",
                        (
                            f"{rates.mean():,.2f}"
                            if not rates.empty
                            else "Unavailable"
                        )
                    )

                    st.caption(
                        "These figures summarize your saved CSV. "
                        "Missing and negative room rates are excluded "
                        "from the average nightly rate."
                    )

                    left, right = st.columns(2)

                    with left:
                        st.subheader(
                            "Cancellation rate by month"
                        )

                        monthly = (
                            outcomes.groupby(
                                "arrival_date_month",
                                as_index=False
                            )["is_canceled"].mean()
                        )

                        monthly["Cancellation rate (%)"] = (
                            monthly["is_canceled"] * 100
                        )

                        fig = px.bar(
                            monthly,
                            x="arrival_date_month",
                            y="Cancellation rate (%)",
                            category_orders={
                                "arrival_date_month": MONTHS
                            },
                            color_discrete_sequence=[
                                "#14b8a6"
                            ],
                            labels={
                                "arrival_date_month":
                                "Arrival month"
                            }
                        )

                        st.plotly_chart(
                            format_chart(fig),
                            width="stretch"
                        )

                    with right:
                        st.subheader(
                            "Booking outcomes by hotel"
                        )

                        counts = (
                            outcomes.groupby(
                                ["hotel", "is_canceled"]
                            )
                            .size()
                            .reset_index(name="Bookings")
                        )

                        counts["Outcome"] = (
                            counts["is_canceled"].map({
                                0: "Not Canceled",
                                1: "Canceled"
                            })
                        )

                        fig = px.bar(
                            counts,
                            x="hotel",
                            y="Bookings",
                            color="Outcome",
                            barmode="group",
                            color_discrete_map={
                                "Not Canceled": "#14b8a6",
                                "Canceled": "#f97373"
                            },
                            labels={"hotel": "Hotel"}
                        )

                        st.plotly_chart(
                            format_chart(fig),
                            width="stretch"
                        )

                    valid_rates = filtered[
                        filtered["adr"].ge(0)
                    ]

                    monthly_rates = (
                        valid_rates.groupby(
                            ["arrival_date_month", "hotel"],
                            as_index=False
                        )["adr"].mean()
                    )

                    fig = px.line(
                        monthly_rates,
                        x="arrival_date_month",
                        y="adr",
                        color="hotel",
                        markers=True,
                        category_orders={
                            "arrival_date_month": MONTHS
                        },
                        labels={
                            "arrival_date_month": "Arrival month",
                            "adr": "Average nightly rate",
                            "hotel": "Hotel"
                        }
                    )

                    

        except Exception as error:
            st.error(
                "The dashboard could not be displayed."
            )

            with st.expander("Show error details"):
                st.code(str(error))


# =========================================================
# PREDICT ONE BOOKING
# =========================================================

elif page == "Predict a Booking":

    show_heading(
        "BOOKING PREDICTION",
        "Check a booking's cancellation risk.",
        "Enter the reservation details to see its prediction "
        "and whether a confirmation follow-up is suggested."
    )

    with st.form("booking_form"):

        reservation_tab, guest_tab, history_tab = st.tabs([
            "Reservation details",
            "Guest details",
            "Booking history"
        ])

        with reservation_tab:
            first, second = st.columns(2)

            with first:
                hotel = st.selectbox(
                    "Hotel",
                    ["Resort Hotel", "City Hotel"]
                )

                arrival = st.date_input(
                    "Arrival date",
                    value=date(2017, 7, 15),
                    help=(
                        "The training data covers 2015–2017. "
                        "Performance on later dates has not "
                        "been established."
                    )
                )

                lead_time = st.number_input(
                    "Days between booking and arrival",
                    min_value=0,
                    value=10,
                    step=1
                )

                room = st.selectbox(
                    "Reserved room type",
                    [
                        "A", "B", "C", "D", "E",
                        "F", "G", "H", "L", "P"
                    ]
                )

            with second:
                weekend_nights = st.number_input(
                    "Weekend nights",
                    min_value=0,
                    value=2,
                    step=1
                )

                week_nights = st.number_input(
                    "Week nights",
                    min_value=0,
                    value=5,
                    step=1
                )

                adr = st.number_input(
                    "Average room price per night",
                    min_value=0.0,
                    value=100.0,
                    step=1.0
                )

                deposit = st.selectbox(
                    "Deposit type",
                    [
                        "No Deposit",
                        "Non Refund",
                        "Refundable"
                    ]
                )

        with guest_tab:
            first, second = st.columns(2)

            with first:
                adults = st.number_input(
                    "Adults",
                    min_value=0,
                    value=2,
                    step=1
                )

                children = st.number_input(
                    "Children",
                    min_value=0,
                    value=0,
                    step=1
                )

                babies = st.number_input(
                    "Babies",
                    min_value=0,
                    value=0,
                    step=1
                )

                country = st.text_input(
                    "Country code",
                    value="USA",
                    help="For example: USA, PRT or ARE."
                )

                meal = st.selectbox(
                    "Meal package",
                    ["BB", "HB", "FB", "SC", "Undefined"]
                )

            with second:
                segment = st.selectbox(
                    "Market segment",
                    [
                        "Online TA",
                        "Offline TA/TO",
                        "Direct",
                        "Corporate",
                        "Groups",
                        "Complementary",
                        "Aviation",
                        "Unknown",
                        "Undefined"
                    ]
                )

                channel = st.selectbox(
                    "Distribution channel",
                    [
                        "TA/TO",
                        "Direct",
                        "Corporate",
                        "GDS",
                        "Undefined"
                    ]
                )

                customer = st.selectbox(
                    "Customer type",
                    [
                        "Transient",
                        "Transient-Party",
                        "Contract",
                        "Group"
                    ]
                )

                agent_input = st.text_input(
                    "Agent ID",
                    value="",
                    help="Leave this blank if the agent is unknown."
                )

        with history_tab:
            repeated = st.selectbox(
                "Is this a returning guest?",
                ["No", "Yes"]
            )

            previous_cancellations = st.number_input(
                "Number of previous cancellations",
                min_value=0,
                value=0,
                step=1
            )

            previous_completed = st.number_input(
                "Previous bookings not canceled",
                min_value=0,
                value=1,
                step=1
            )

        submitted = st.form_submit_button(
            "Predict cancellation",
            type="primary"
        )

    if submitted:
        st.session_state.pop("booking_result", None)

        if adults + children + babies == 0:
            st.error(
                "Enter at least one guest."
            )

        elif weekend_nights + week_nights == 0:
            st.error(
                "Enter at least one night."
            )

        elif check_model():
            try:
                agent = prepare_agent(
                    agent_input,
                    classifier
                )

                booking = pd.DataFrame({
                    "hotel": [hotel],
                    "lead_time": [lead_time],
                    "arrival_date_month": [
                        arrival.strftime("%B")
                    ],
                    "arrival_date_year": [arrival.year],
                    "arrival_date_day_of_month": [
                        arrival.day
                    ],
                    "stays_in_weekend_nights": [
                        weekend_nights
                    ],
                    "stays_in_week_nights": [
                        week_nights
                    ],
                    "adults": [adults],
                    "children": [children],
                    "babies": [babies],
                    "meal": [meal],
                    "country": [
                        country.strip().upper()
                        if country.strip()
                        else "Unknown"
                    ],
                    "market_segment": [segment],
                    "distribution_channel": [channel],
                    "is_repeated_guest": [
                        int(repeated == "Yes")
                    ],
                    "previous_cancellations": [
                        previous_cancellations
                    ],
                    "previous_bookings_not_canceled": [
                        previous_completed
                    ],
                    "reserved_room_type": [room],
                    "deposit_type": [deposit],
                    "agent": [agent],
                    "customer_type": [customer],
                    "adr": [adr]
                })

                expected = list(
                    getattr(
                        classifier,
                        "feature_names_in_",
                        FEATURES
                    )
                )

                missing = (
                    set(expected) - set(booking.columns)
                )

                if missing:
                    raise ValueError(
                        "The saved model expects different inputs: "
                        + ", ".join(sorted(missing))
                    )

                booking = booking[expected]

                with st.spinner(
                    "Calculating the prediction..."
                ):
                    prediction = classifier.predict(
                        booking
                    )[0]

                    classes = list(
                        classifier.classes_
                    )

                    if 1 not in classes:
                        raise ValueError(
                            "The saved model must use 1 "
                            "for canceled bookings."
                        )

                    probability = (
                        classifier.predict_proba(booking)[
                            0, classes.index(1)
                        ]
                    )

                st.session_state["booking_result"] = {
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "booking": booking,
                    "nights": (
                        weekend_nights + week_nights
                    ),
                    "guests": (
                        adults + children + babies
                    ),
                    "room_charges": (
                        weekend_nights + week_nights
                    ) * adr
                }

            except Exception as error:
                st.error(
                    "The prediction could not be completed."
                )

                with st.expander("Show error details"):
                    st.code(str(error))

    if "booking_result" in st.session_state:
        result = st.session_state["booking_result"]

        probability = result["probability"]
        threshold_value = threshold / 100

        st.divider()
        st.subheader("Prediction result")

        left, right = st.columns([1, 1.3])

        with left:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={
                        "text": "Estimated cancellation probability"
                    },
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#234565"},
                        "steps": [
                            {
                                "range": [0, threshold],
                                "color": "#d5f4eb"
                            },
                            {
                                "range": [threshold, 100],
                                "color": "#ffdede"
                            }
                        ],
                        "threshold": {
                            "line": {
                                "color": "#ef4444",
                                "width": 3
                            },
                            "value": threshold
                        }
                    }
                )
            )

            gauge.update_layout(height=300)

            st.plotly_chart(
                format_chart(gauge),
                width="stretch"
            )

        with right:
            if result["prediction"] == 1:
                st.markdown(
                    "### Model prediction: likely to cancel"
                )
            else:
                st.markdown(
                    "### Model prediction: likely not to cancel"
                )

            if probability >= threshold_value:
                st.warning(
                    "A confirmation follow-up is suggested "
                    "for this booking."
                )
            else:
                st.success(
                    "This booking is not flagged for priority "
                    "follow-up. Continue with the usual "
                    "confirmation process."
                )

            st.write(
                f"Your selected follow-up threshold is "
                f"**{threshold}%**."
            )

            st.caption(
                "The threshold controls the follow-up suggestion. "
                "It does not change the model's original prediction."
            )

            st.caption(
                "The percentage is an estimate, "
                "not a guarantee that the guest will cancel."
            )

        first, second, third = st.columns(3)

        first.metric(
            "Total nights",
            result["nights"]
        )

        second.metric(
            "Total guests",
            result["guests"]
        )

        third.metric(
            "Estimated room charges",
            f"{result['room_charges']:,.2f}"
        )

        st.caption(
            "Estimated room charges = nightly rate × nights. "
            "Extras and taxes are not included."
        )

        with st.expander("View the booking details"):
            st.dataframe(
                result["booking"],
                width="stretch",
                hide_index=True
            )

        st.caption(
            "Showing the last submitted booking. "
            "After changing any details, "
            "click Predict cancellation again."
        )


# =========================================================
# ABOUT THE MODEL
# =========================================================

else:
    show_heading(
        "ABOUT THIS PROJECT",
        "How the prediction works.",
        "Understand the business question, "
        "model results and limitations."
    )

    st.subheader("Business question")

    st.write(
        "Which hotel bookings are likely to be canceled, "
        "so hotel staff can prioritize confirmation follow-ups "
        "and improve room availability planning?"
    )

    st.subheader("Selected model")

    st.write(
        "Random Forest was selected because it achieved "
        "the highest average accuracy and cancellation "
        "F1-score in the recorded cross-validation comparison."
    )

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "KNN",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            0.7528, 0.7566, 0.7663, 0.7793
        ],
        "Precision": [
            0.7786, 0.6968, 0.7728, 0.7562
        ],
        "Recall": [
            0.4683, 0.6111, 0.5262, 0.5997
        ],
        "F1": [
            0.5848, 0.6512, 0.6260, 0.6689
        ]
    })

    first, second, third = st.columns(3)

    first.metric(
        "Recorded RF validation accuracy",
        "77.93%"
    )

    second.metric(
        "Cancellation recall",
        "59.97%"
    )

    third.metric(
        "Cancellation F1-score",
        "66.89%"
    )

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1": "{:.2%}"
        }),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "These are the five-fold cross-validation results "
        "recorded in the project notebook. "
        "They are not recalculated by this app."
    )

    chart_data = comparison.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Recall", "F1"],
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        chart_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=[
            "#234565", "#14b8a6", "#e9b44c"
        ]
    )

    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%"
    )

    st.plotly_chart(
        format_chart(fig),
        width="stretch"
    )

    with st.expander("What do the scores mean?"):
        st.markdown(
            """
            - **Accuracy:** how many bookings were classified
              correctly overall.
            - **Precision:** how many cancellation alerts
              were correct.
            - **Recall:** how many actual cancellations
              the model identified.
            - **F1-score:** a score combining precision
              and recall.
            """
        )

    with st.expander("What does the follow-up threshold do?"):
        st.write(
            "The threshold decides when the app suggests "
            "a confirmation follow-up."
        )

        st.write(
            "For example, at a 50% threshold, "
            "a booking with an estimated 70% cancellation "
            "probability is flagged for follow-up. "
            "A booking at 30% is not."
        )

        st.write(
            "Lowering the threshold flags more bookings. "
            "Raising it flags fewer bookings."
        )

    with st.expander("Project limitations"):
        st.markdown(
            """
            - The dataset contains original and synthetic bookings.
            - The model showed overfitting during evaluation.
            - Some cancellations will not be identified.
            - Probability estimates are uncertain and have
              not been verified as calibrated probabilities.
            - Historical results do not establish performance
              for future dates or new hotels.
            - A final evaluation should use original bookings
              excluded from synthetic-data generation.
            - Follow-up suggestions support staff decisions;
              the app does not contact guests automatically.
            """
        )

    with st.expander("Which information does the model use?"):
        st.dataframe(
            pd.DataFrame({
                "Input column": FEATURES
            }),
            width="stretch",
            hide_index=True
        )


# Add a short footer.
st.markdown(
    """
    <div class="footer">
        StayWise · Hotel Booking Prediction<br>
        Data Science for Business Applications 
    </div>
    """,
    unsafe_allow_html=True
)
