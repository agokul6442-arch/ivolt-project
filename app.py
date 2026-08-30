import os
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Set base directories and file paths
PROJECT_DIR = Path(__file__).resolve().parent
CSV_FILE = PROJECT_DIR / "EV-MODEL-DATASET20.csv"
MODEL_FILE = PROJECT_DIR / "ridge_ev_range_model.pkl"

# CSV loading with fast optimization
def load_csv_fast(file_path):
    """
    Load CSV file with fast optimization.
    Uses pandas efficient settings for quick data loading.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        DataFrame if successful, empty DataFrame if fails
    """
    try:
        # Optimized pandas reading - NO delays!
        print(f"Loading CSV: {file_path}")
        df = pd.read_csv(
            file_path,
            engine='c',              # Fast C engine
            dtype={'Car_RegNo': str, 'Driver_ID': str}  # Specify dtypes to avoid inference
        )
        print(f"✓ CSV loaded in milliseconds: {len(df)} records")
        return df
    except FileNotFoundError:
        print(f"✗ CSV file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return pd.DataFrame()

# Sklearn unpickling patch for legacy serialized column transformer lists
try:
    import sklearn.compose._column_transformer as column_transformer
    column_transformer._RemainderColsList = type(
        "_RemainderColsList", (list,), {}
    )
except Exception:
    pass

# Safe loading of ML Model
try:
    range_model = joblib.load(MODEL_FILE)
except Exception as e:
    print(f"Warning: Could not load model {MODEL_FILE}. Error: {e}")
    range_model = None

# Function to get fresh CSV data from disk (dynamic)
def get_csv_dataframe():
    """Load CSV fresh from disk every time it's called."""
    return load_csv_fast(CSV_FILE)


def refresh_latest_vehicle_df(dataframe=None):
    """Get latest vehicle records from dataframe (reloads from disk if needed)."""
    if dataframe is None:
        dataframe = get_csv_dataframe()
    
    if dataframe.empty:
        return dataframe.copy()

    latest = dataframe.copy()
    if "Observation_DateTime" in latest.columns:
        latest["_observation_time"] = pd.to_datetime(
            latest["Observation_DateTime"], dayfirst=True, errors="coerce"
        )
        latest = latest.sort_values("_observation_time")
    latest = latest.drop_duplicates(subset=["Car_RegNo"], keep="last") if "Car_RegNo" in latest.columns else latest
    return latest.reset_index(drop=True)


def latest_vehicle_rows(dataframe):
    if dataframe.empty or "Car_RegNo" not in dataframe.columns:
        return dataframe

    latest = dataframe.copy()
    if "Observation_DateTime" in latest.columns:
        latest["_observation_time"] = pd.to_datetime(
            latest["Observation_DateTime"], dayfirst=True, errors="coerce"
        )
        latest = latest.sort_values("_observation_time")
    return latest.drop_duplicates(subset=["Car_RegNo"], keep="last")


def clamp_soc(value):
    try:
        return min(100, max(1, float(value)))
    except (TypeError, ValueError):
        return 1


def get_latest_vehicle_rows():
    """Get latest vehicle rows by reloading CSV from disk."""
    dataframe = get_csv_dataframe()
    return refresh_latest_vehicle_df(dataframe)


# ==========================
# HOME & STATIC PAGES
# ==========================
@app.route("/")
def home():
    return send_from_directory(PROJECT_DIR, "Homepage.html")


@app.route("/<path:page>")
def static_page(page):
    """Serve HTML pages from the project directory."""
    if page.endswith(".html"):
        return send_from_directory(PROJECT_DIR, page)
    return jsonify({"error": "Page not found"}), 404


# ==========================
# ADMIN DASHBOARD API
# ==========================
@app.route("/api/admin")
def admin_dashboard():
    dataframe = get_csv_dataframe()
    if dataframe.empty:
        return jsonify({
            "totalCars": 0,
            "workingCars": 0,
            "chargingCars": 0,
            "runningCars": 0,
            "totalRevenue": 0
        })

    vehicle_rows = latest_vehicle_rows(dataframe)
    total_cars = len(vehicle_rows)

    charging_cars = len(
        vehicle_rows[vehicle_rows["Charging_Status"].astype(str).str.lower().isin(["yes", "charging"])]
    ) if "Charging_Status" in dataframe.columns else 0

    running_cars = len(
        vehicle_rows[vehicle_rows["Running_Status"].astype(str).str.lower().isin(["yes", "running"])]
    ) if "Running_Status" in dataframe.columns else 0

    working_cars = len(
        vehicle_rows[vehicle_rows["Working"].astype(str).str.lower() == "working"]
    ) if "Working" in dataframe.columns else running_cars + charging_cars

    total_revenue = (
        float(vehicle_rows["Per_Day_Revenue"].sum())
        if "Per_Day_Revenue" in dataframe.columns
        else 0
    )

    return jsonify({
        "totalCars": int(total_cars),
        "workingCars": int(working_cars),
        "chargingCars": int(charging_cars),
        "runningCars": int(running_cars),
        "totalRevenue": round(total_revenue, 2)
    })


# ==========================
# VEHICLE & DRIVER APIS
# ==========================
@app.route("/api/vehicles")
def vehicles():
    dataframe = get_csv_dataframe()
    if dataframe.empty:
        return jsonify([])
    latest = refresh_latest_vehicle_df(dataframe)
    return jsonify(latest.fillna("").to_dict(orient="records"))


@app.route("/api/drivers")
def drivers():
    dataframe = get_csv_dataframe()
    if dataframe.empty or "Driver_Name" not in dataframe.columns:
        return jsonify([])

    updated_df = refresh_latest_vehicle_df(dataframe)
    drivers_list = updated_df["Driver_Name"].dropna().unique().tolist()
    return jsonify(drivers_list)


@app.route("/api/vehicle/<reg_no>")
def vehicle_details(reg_no):
    dataframe = get_csv_dataframe()
    if dataframe.empty or "Car_RegNo" not in dataframe.columns:
        return jsonify({"error": "Car_RegNo column not found"})

    vehicle = dataframe[dataframe["Car_RegNo"].astype(str).str.upper() == reg_no.upper()]

    if vehicle.empty:
        return jsonify({"error": "Vehicle not found"})

    vehicle_data = vehicle.iloc[0].fillna("").to_dict()
    if "SOC_Percentage" in vehicle_data:
        vehicle_data["SOC_Percentage"] = clamp_soc(vehicle_data["SOC_Percentage"])
    return jsonify(vehicle_data)


@app.route("/api/driver/<reg_no>")
def driver_dashboard(reg_no):
    dataframe = get_csv_dataframe()
    if dataframe.empty or "Car_RegNo" not in dataframe.columns:
        return jsonify({"error": "Car_RegNo column not found"})

    vehicle = dataframe[dataframe["Car_RegNo"].astype(str).str.upper() == reg_no.upper()]

    if vehicle.empty:
        return jsonify({"error": "Vehicle not found"})

    row = vehicle.iloc[0]

    return jsonify({
        "registrationNumber": str(row.get("Car_RegNo", "")),
        "vehicleName": str(row.get("Car_Name", "")),
        "driverName": str(row.get("Driver_Name", "")),
        "soc": clamp_soc(row.get("SOC_Percentage", 0)),
        "batteryCapacity": float(row.get("Battery_Capacity_kWh", 0)),
        "range": float(row.get("Estimated Range_Range", 0)),
        "chargingStatus": str(row.get("Charging_Status", "")),
        "runningStatus": str(row.get("Running_Status", "")),
        "location": str(row.get("Location", "")),
        "destination": str(row.get("Destination_Location", "")),
        "tripRevenue": float(row.get("Trip_Revenue", 0)),
        "dailyRevenue": float(row.get("Per_Day_Revenue", 0))
    })


# ==========================
# REVENUE & SOC APIS
# ==========================
@app.route("/api/revenue")
def revenue():
    dataframe = get_csv_dataframe()
    if dataframe.empty or "Per_Day_Revenue" not in dataframe.columns:
        return jsonify([])

    latest = refresh_latest_vehicle_df(dataframe)
    revenue_data = [
        {"vehicle": row.get("Car_Name", ""), "revenue": row.get("Per_Day_Revenue", 0)}
        for _, row in latest.iterrows()
    ]
    return jsonify(revenue_data)


@app.route("/api/soc")
def soc_chart():
    dataframe = get_csv_dataframe()
    if dataframe.empty:
        return jsonify([])

    latest = refresh_latest_vehicle_df(dataframe)
    chart = [
        {"vehicle": row.get("Car_Name", ""), "soc": clamp_soc(row.get("SOC_Percentage", 0))}
        for _, row in latest.iterrows()
    ]
    return jsonify(chart)


# ==========================
# RIDGE EV RANGE PREDICTION API
# ==========================
@app.route("/api/predict-range", methods=["POST"])
def predict_range():
    if range_model is None:
        return jsonify({"error": "Prediction model is not loaded"}), 500

    data = request.get_json(silent=True) or {}
    feature_names = [
        "SOC_Percentage", "Battery_Capacity_kWh", "Max_Range_km", "Top Speed",
        "Acceleration 0 - 100 km/h", "Motor_Power_kW", "Motor_Torque_Nm",
        "EV_Weight_kg", "Passenger_Count",
        "Energy_Consumption_kWh_per_100km", "Highway_City"
    ]

    try:
        features = {name: data[name] for name in feature_names}
        prediction = float(range_model.predict(pd.DataFrame([features]))[0])
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": f"Invalid prediction inputs: {error}"}), 400

    return jsonify({"predictedRangeKm": round(max(0, prediction), 1)})


# ==========================
# AUTH & REGISTRATION APIS
# ==========================
@app.route("/api/admin-login", methods=["POST"])
def admin_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")

    if username == "admin" and password == "admin123":
        return jsonify({"success": True, "message": "Admin login successful"})

    return jsonify({"success": False, "message": "Invalid username or password"})


@app.route("/api/register", methods=["POST"])
def register_driver():
    data = request.get_json(silent=True) or {}
    required_fields = [
        "fullName", "age", "experience", "email", "phone",
        "license", "carRegistration", "password"
    ]

    if any(not str(data.get(field, "")).strip() for field in required_fields):
        return jsonify({"success": False, "message": "All registration fields are required."}), 400

    if data.get("password") != data.get("confirmPassword"):
        return jsonify({"success": False, "message": "Passwords do not match."}), 400

    # Load fresh CSV
    df = get_csv_dataframe()
    license_number = str(data["license"]).strip()
    phone_number = str(data["phone"]).strip()
    car_registration = str(data["carRegistration"]).strip().upper()

    if not df.empty:
        duplicate_license = df["Driver_Licence_Number"].astype(str).str.strip().str.upper() == license_number.upper()
        duplicate_phone = df["Driver_MobileNo"].astype(str).str.strip() == phone_number
        duplicate_car = df["Car_RegNo"].astype(str).str.strip().str.upper() == car_registration

        if duplicate_license.any():
            return jsonify({"success": False, "message": "This driving license is already registered."}), 409
        if duplicate_phone.any():
            return jsonify({"success": False, "message": "This phone number is already registered."}), 409
        if duplicate_car.any():
            return jsonify({"success": False, "message": "This car registration is already registered."}), 409

        existing_ids = pd.to_numeric(
            df["Driver_ID"].astype(str).str.extract(r"(\d+)$")[0], errors="coerce"
        ).dropna()
        next_id = int(existing_ids.max()) + 1 if not existing_ids.empty else 1
    else:
        next_id = 1

    driver_id = f"DRV{next_id:03d}"

    new_record = {column: "" for column in df.columns} if not df.empty else {}
    new_record.update({
        "Observation_ID": f"REG{next_id:05d}",
        "Car_RegNo": car_registration,
        "Driver_ID": driver_id,
        "Driver_Name": str(data["fullName"]).strip(),
        "Driver_Licence_Number": license_number,
        "Driver_MobileNo": phone_number,
        "Driver_Email": str(data["email"]).strip(),
        "Driver_Password": str(data["password"]),
        "Confirm_Driver_Password": str(data["confirmPassword"]),
        "Driver_Age": int(data["age"]),
        "Driver_Years_of_Experience": int(data["experience"]),
        "Working": "Working",
        "Charging_Status": "No",
        "Running_Status": "No",
        "SOC_Percentage": 100,
        "Per_Day_Revenue": 0,
        "Trip_Revenue": 0
    })

    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

    return jsonify({
        "success": True,
        "message": "Driver account created successfully.",
        "driver_id": driver_id
    }), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")

    # Load fresh CSV
    df = get_csv_dataframe()
    if df.empty:
        return jsonify({"success": False})

    user = df[
        (df["Driver_ID"].astype(str) == str(username)) &
        (df["Driver_Password"].astype(str) == str(password))
    ]

    if not user.empty:
        row = user.iloc[0]
        return jsonify({
            "success": True,
            "driver_id": row["Driver_ID"],
            "car_regno": row["Car_RegNo"]
        })

    return jsonify({"success": False})


@app.route("/api/driverdetails/<driver_id>")
def driver_details(driver_id):
    # Load fresh CSV
    df = get_csv_dataframe()
    if df.empty:
        return jsonify({"error": "Driver not found"})

    user = df[df["Driver_ID"].astype(str) == str(driver_id)]

    if user.empty:
        return jsonify({"error": "Driver not found"})

    driver_data = user.iloc[0].fillna("").to_dict()
    if "SOC_Percentage" in driver_data:
        driver_data["SOC_Percentage"] = clamp_soc(driver_data["SOC_Percentage"])
    return jsonify(driver_data)


# ==========================
# RUN APP LOCAL DEVELOPMENT
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
