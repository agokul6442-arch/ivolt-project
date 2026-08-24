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

# Sklearn unpickling patch for legacy serialized column transformer lists
try:
    import sklearn.compose._column_transformer as column_transformer
    column_transformer._RemainderColsList = type(
        "_RemainderColsList", (list,), {}
    )
except Exception:
    pass

# Safe loading of Dataset
try:
    df = pd.read_csv(CSV_FILE)
except Exception as e:
    print(f"Warning: Could not load dataset {CSV_FILE}. Error: {e}")
    df = pd.DataFrame()

# Safe loading of ML Model
try:
    range_model = joblib.load(MODEL_FILE)
except Exception as e:
    print(f"Warning: Could not load model {MODEL_FILE}. Error: {e}")
    range_model = None


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
    if df.empty:
        return jsonify({
            "totalCars": 0,
            "workingCars": 0,
            "chargingCars": 0,
            "runningCars": 0,
            "totalRevenue": 0
        })

    vehicle_rows = latest_vehicle_rows(df)
    total_cars = len(vehicle_rows)

    charging_cars = len(
        vehicle_rows[vehicle_rows["Charging_Status"].astype(str).str.lower().isin(["yes", "charging"])]
    ) if "Charging_Status" in df.columns else 0

    running_cars = len(
        vehicle_rows[vehicle_rows["Running_Status"].astype(str).str.lower().isin(["yes", "running"])]
    ) if "Running_Status" in df.columns else 0

    working_cars = len(
        vehicle_rows[vehicle_rows["Working"].astype(str).str.lower() == "working"]
    ) if "Working" in df.columns else running_cars + charging_cars

    total_revenue = (
        float(vehicle_rows["Per_Day_Revenue"].sum())
        if "Per_Day_Revenue" in df.columns
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
    if df.empty:
        return jsonify([])
    return jsonify(df.fillna("").to_dict(orient="records"))


@app.route("/api/drivers")
def drivers():
    if df.empty or "Driver_Name" not in df.columns:
        return jsonify([])

    drivers_list = df["Driver_Name"].dropna().unique().tolist()
    return jsonify(drivers_list)


@app.route("/api/vehicle/<reg_no>")
def vehicle_details(reg_no):
    if df.empty or "Car_RegNo" not in df.columns:
        return jsonify({"error": "Car_RegNo column not found"})

    vehicle = df[df["Car_RegNo"].astype(str).str.upper() == reg_no.upper()]

    if vehicle.empty:
        return jsonify({"error": "Vehicle not found"})

    return jsonify(vehicle.iloc[0].fillna("").to_dict())


@app.route("/api/driver/<reg_no>")
def driver_dashboard(reg_no):
    if df.empty or "Car_RegNo" not in df.columns:
        return jsonify({"error": "Car_RegNo column not found"})

    vehicle = df[df["Car_RegNo"].astype(str).str.upper() == reg_no.upper()]

    if vehicle.empty:
        return jsonify({"error": "Vehicle not found"})

    row = vehicle.iloc[0]

    return jsonify({
        "registrationNumber": str(row.get("Car_RegNo", "")),
        "vehicleName": str(row.get("Car_Name", "")),
        "driverName": str(row.get("Driver_Name", "")),
        "soc": float(row.get("SOC_Percentage", 0)),
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
    if df.empty or "Per_Day_Revenue" not in df.columns:
        return jsonify([])

    revenue_data = [
        {"vehicle": row.get("Car_Name", ""), "revenue": row.get("Per_Day_Revenue", 0)}
        for _, row in df.iterrows()
    ]
    return jsonify(revenue_data)


@app.route("/api/soc")
def soc_chart():
    if df.empty:
        return jsonify([])

    chart = [
        {"vehicle": row.get("Car_Name", ""), "soc": row.get("SOC_Percentage", 0)}
        for _, row in df.iterrows()
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
    global df

    data = request.get_json(silent=True) or {}
    required_fields = [
        "fullName", "age", "experience", "email", "phone",
        "license", "carRegistration", "password"
    ]

    if any(not str(data.get(field, "")).strip() for field in required_fields):
        return jsonify({"success": False, "message": "All registration fields are required."}), 400

    if data.get("password") != data.get("confirmPassword"):
        return jsonify({"success": False, "message": "Passwords do not match."}), 400

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
    if df.empty:
        return jsonify({"error": "Driver not found"})

    user = df[df["Driver_ID"].astype(str) == str(driver_id)]

    if user.empty:
        return jsonify({"error": "Driver not found"})

    return jsonify(user.iloc[0].fillna("").to_dict())


# ==========================
# RUN APP LOCAL DEVELOPMENT
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)