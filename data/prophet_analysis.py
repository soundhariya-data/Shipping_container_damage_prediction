import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# PORT COORDINATES — for India map
# ═══════════════════════════════════════════════════════════════
PORT_COORDINATES = {
    'Mumbai'      : (19.0760, 72.8777),
    'Nhava Sheva' : (18.9500, 72.9500),
    'Chennai'     : (13.0827, 80.2707),
    'Vizag'       : (17.6868, 83.2185),
    'Kolkata'     : (22.5726, 88.3639),
    'Kochi'       : (9.9312,  76.2673),
    'Mundra'      : (22.8394, 69.7064),
    'Kandla'      : (23.0333, 70.2167),
    'Pipavav'     : (20.9167, 71.5167),
    'Paradip'     : (20.3167, 86.6167),
    'Ennore'      : (13.2167, 80.3167),
    'Tuticorin'   : (8.7642,  78.1348),
    'Haldia'      : (22.0667, 88.1000),
    'Deendayal'   : (23.0333, 70.2167),
    'Goa'         : (15.4909, 73.8278),
    'New Mangalore': (12.9141, 74.8560),
    # International ports
    'Colombo'     : (6.9271,  79.8612),
    'Singapore'   : (1.3521,  103.8198),
    'Shanghai'    : (31.2304, 121.4737),
    'Dubai'       : (25.2048, 55.2708),
    'Jebel Ali'   : (24.9857, 55.0272),
    'Abu Dhabi'   : (24.4539, 54.3773),
    'Port Klang'  : (3.0000,  101.4000),
}


# ═══════════════════════════════════════════════════════════════
# FUNCTION 1 — Run Prophet for ONE variable
# ═══════════════════════════════════════════════════════════════
def run_prophet_for_variable(one_container, variable,
                             forecast_days=60):
    """
    Runs Prophet on any variable (shock_g or wave_m).
    Returns historical results with anomaly flags
    and future forecast.
    """
    # Prepare data
    prophet_df = one_container[['timestamp', variable]].rename(
        columns={'timestamp': 'ds', variable: 'y'}
    ).dropna()

    # Train Prophet
    model = Prophet(
        interval_width     = 0.95,
        daily_seasonality  = False,
        weekly_seasonality = True,
        yearly_seasonality = True
    )
    model.fit(prophet_df)

    # Forecast
    future   = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Split historical vs future
    last_date  = prophet_df['ds'].max()
    historical = forecast[forecast['ds'] <= last_date].copy()
    historical['actual'] = prophet_df['y'].values
    historical['anomaly'] = (
        (historical['actual'] > historical['yhat_upper']) |
        (historical['actual'] < historical['yhat_lower'])
    )

    future_fc = forecast[forecast['ds'] > last_date].copy()
    future_fc['actual']  = None
    future_fc['anomaly'] = False

    return historical, future_fc, model


# ═══════════════════════════════════════════════════════════════
# FUNCTION 2 — Run Full Analysis for ONE container
# ═══════════════════════════════════════════════════════════════
def run_prophet(df, container_id, forecast_days=60):
    """
    Runs Prophet for shock_g AND wave_m for one container.
    Returns historical + future for both variables.
    """
    one_container = df[df['containerID'] == container_id].copy()
    one_container = one_container.reset_index()

    if 'timestamp' not in one_container.columns:
        one_container = one_container.rename(
            columns={'index': 'timestamp'}
        )
    one_container['timestamp'] = pd.to_datetime(
        one_container['timestamp']
    )

    # Prophet for shock_g
    shock_hist, shock_future, _ = run_prophet_for_variable(
        one_container, 'shock_g', forecast_days
    )

    # Add metadata
    shock_hist['containerID'] = container_id
    shock_hist['damaged']     = one_container['damaged(T)'].values
    shock_hist['wave_m']      = one_container['wave_m'].values
    shock_hist['wind_kph']    = one_container['wind_kph'].values
    shock_hist['storm']       = one_container['storm'].values
    shock_hist['route']       = one_container['route'].values
    shock_hist['final_port']  = one_container['final_port'].values
    shock_hist['cargo_type']  = one_container['type'].values
    shock_future['containerID'] = container_id

    # Prophet for wave_m
    wave_hist, wave_future, _ = run_prophet_for_variable(
        one_container, 'wave_m', forecast_days
    )
    wave_hist['containerID']   = container_id
    wave_future['containerID'] = container_id

    return (
        shock_hist, shock_future,
        wave_hist,  wave_future,
        one_container
    )


# ═══════════════════════════════════════════════════════════════
# FUNCTION 3 — Weather Risk Score
# ═══════════════════════════════════════════════════════════════
def weather_risk_score(wave_m, wind_kph, storm):
    """
    Returns (score 0-100, label)
    Wave=40pts, Wind=30pts, Storm=30pts
    """
    if   wave_m >= 4.0: wave_risk = 40
    elif wave_m >= 3.0: wave_risk = 30
    elif wave_m >= 2.0: wave_risk = 20
    elif wave_m >= 1.5: wave_risk = 10
    else:               wave_risk = 5

    if   wind_kph >= 30: wind_risk = 30
    elif wind_kph >= 25: wind_risk = 20
    elif wind_kph >= 20: wind_risk = 15
    elif wind_kph >= 15: wind_risk = 10
    else:                wind_risk = 5

    storm_map = {
        'Green': 5, 'Yellow': 15,
        'Orange': 25, 'Red': 30,
        '0': 0, '1': 30
    }
    storm_risk = storm_map.get(str(storm), 10)
    total      = wave_risk + wind_risk + storm_risk

    if   total >= 75: label = "🔴 Critical"
    elif total >= 50: label = "🟠 High"
    elif total >= 30: label = "🟡 Medium"
    else:             label = "🟢 Low"

    return total, label


# ═══════════════════════════════════════════════════════════════
# FUNCTION 4 — Port Risk Score
# ═══════════════════════════════════════════════════════════════
def port_risk_score(df):
    """
    Returns port risk table sorted by risk score.
    damage_rate = damaged/total × 100
    risk_score  = damage_rate × 3 (max 100)
    """
    port_stats = df.groupby('final_port').agg(
        total   = ('damaged(T)', 'count'),
        damaged = ('damaged(T)', 'sum')
    ).reset_index()

    port_stats['damage_rate'] = (
        port_stats['damaged'] / port_stats['total'] * 100
    ).round(2)

    port_stats['risk_score'] = (
        port_stats['damage_rate'] * 3
    ).clip(upper=100).round(1)

    def get_label(score):
        if   score >= 75: return "🔴 Critical"
        elif score >= 50: return "🟠 High"
        elif score >= 25: return "🟡 Medium"
        else:             return "🟢 Low"

    port_stats['risk_label'] = port_stats['risk_score'].apply(
        get_label
    )
    # Add coordinates
    port_stats['lat'] = port_stats['final_port'].map(
        lambda p: PORT_COORDINATES.get(p, (None, None))[0]
    )
    port_stats['lon'] = port_stats['final_port'].map(
        lambda p: PORT_COORDINATES.get(p, (None, None))[1]
    )

    return port_stats.sort_values(
        'risk_score', ascending=False
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# FUNCTION 5 — Get Risk for ONE Future Date
# ═══════════════════════════════════════════════════════════════
def get_future_date_risk(shock_future, wave_future,
                         port_scores, selected_date,
                         container_ports):
    """
    Returns predicted risk for a specific future date.
    """
    selected_date = pd.Timestamp(selected_date)

    shock_row = shock_future[
        shock_future['ds'].dt.date == selected_date.date()
    ]
    wave_row = wave_future[
        wave_future['ds'].dt.date == selected_date.date()
    ]

    if shock_row.empty or wave_row.empty:
        return None

    predicted_shock = shock_row['yhat'].values[0]
    shock_upper     = shock_row['yhat_upper'].values[0]
    shock_lower     = shock_row['yhat_lower'].values[0]
    predicted_wave  = wave_row['yhat'].values[0]
    wave_upper      = wave_row['yhat_upper'].values[0]

    # Weather risk
    weather_score, weather_label = weather_risk_score(
        predicted_wave, 15, 'Green'
    )

    # Port risk
    port_risk_dict = dict(zip(
        port_scores['final_port'],
        port_scores['risk_score']
    ))
    port_risks    = [
        port_risk_dict.get(p, 0) for p in container_ports
    ]
    max_port_risk = max(port_risks) if port_risks else 0
    riskiest_port = (
        container_ports[np.argmax(port_risks)]
        if len(container_ports) > 0 else "Unknown"
    )

    # Fixed flag logic
    shock_band     = shock_upper - shock_lower
    shock_position = (
        (predicted_shock - shock_lower) / shock_band
        if shock_band > 0 else 0
    )
    shock_flag   = (
        shock_position > 0.60 or
        predicted_shock > 1.8
    )
    weather_flag = weather_score > 25
    port_flag    = max_port_risk > 25

    flags = sum([shock_flag, weather_flag, port_flag])
    if   flags == 3: overall = "🔴 Critical"
    elif flags == 2: overall = "🟠 High"
    elif flags == 1: overall = "🟡 Medium"
    else:            overall = "🟢 Low"

    return {
        'date'           : selected_date,
        'predicted_shock': round(predicted_shock, 3),
        'shock_upper'    : round(shock_upper, 3),
        'shock_lower'    : round(shock_lower, 3),
        'shock_position' : round(shock_position * 100, 1),
        'predicted_wave' : round(predicted_wave, 3),
        'wave_upper'     : round(wave_upper, 3),
        'weather_score'  : weather_score,
        'weather_label'  : weather_label,
        'riskiest_port'  : riskiest_port,
        'max_port_risk'  : max_port_risk,
        'shock_flag'     : shock_flag,
        'weather_flag'   : weather_flag,
        'port_flag'      : port_flag,
        'overall_risk'   : overall,
        'flags'          : flags
    }


# ═══════════════════════════════════════════════════════════════
# FUNCTION 6 — Parse Routes into Port Lists
# ═══════════════════════════════════════════════════════════════
def parse_route_ports(route_str):
    """
    Splits 'Mumbai-Kochi-Colombo' into
    ['Mumbai', 'Kochi', 'Colombo']
    """
    if pd.isna(route_str):
        return []
    return [p.strip() for p in route_str.split('-')]


# ═══════════════════════════════════════════════════════════════
# FUNCTION 7 — Get Route Data for Map
# ═══════════════════════════════════════════════════════════════
def get_route_map_data(df, container_id=None):
    """
    Returns route segments with coordinates
    for plotting on the India map.
    If container_id given, returns only that container's routes.
    """
    if container_id:
        sub = df[df['containerID'] == container_id].copy()
    else:
        sub = df.copy()

    routes = sub['route'].unique()
    segments = []

    for route in routes:
        ports = parse_route_ports(route)
        for i in range(len(ports) - 1):
            p1 = ports[i]
            p2 = ports[i + 1]
            if (p1 in PORT_COORDINATES and
                    p2 in PORT_COORDINATES):
                lat1, lon1 = PORT_COORDINATES[p1]
                lat2, lon2 = PORT_COORDINATES[p2]
                # Get containers on this route
                containers_on_route = sub[
                    sub['route'] == route
                ]['containerID'].unique().tolist()
                # Get damage count on this route
                damage_on_route = int(sub[
                    sub['route'] == route
                ]['damaged(T)'].sum())
                segments.append({
                    'from'        : p1,
                    'to'          : p2,
                    'route'       : route,
                    'lat1'        : lat1,
                    'lon1'        : lon1,
                    'lat2'        : lat2,
                    'lon2'        : lon2,
                    'containers'  : containers_on_route,
                    'damage_count': damage_on_route
                })

    return pd.DataFrame(segments)


# ═══════════════════════════════════════════════════════════════
# FUNCTION 8 — Fleet Wide Analysis
# ═══════════════════════════════════════════════════════════════
def fleet_wide_analysis(df, forecast_days=60):
    """
    Runs full analysis for ALL containers.
    Returns fleet summary + port scores.
    """
    port_scores    = port_risk_score(df)
    port_risk_dict = dict(zip(
        port_scores['final_port'],
        port_scores['risk_score']
    ))

    fleet_results = []

    for container_id in df['containerID'].unique():
        print(f"  Analyzing {container_id}...")

        (shock_hist, shock_future,
         wave_hist, wave_future,
         one_container) = run_prophet(
            df, container_id, forecast_days
        )
        anomalies = shock_hist[shock_hist['anomaly'] == True]

        # Condition 1 — Shock
        shock_count = len(anomalies)
        max_shock   = shock_hist['actual'].max()
        if   shock_count >= 5: shock_level = "🔴 Critical"
        elif shock_count >= 2: shock_level = "🟠 High"
        elif shock_count >= 1: shock_level = "🟡 Medium"
        else:                  shock_level = "🟢 Low"
        shock_flag = shock_count >= 1

        # Condition 2 — Weather
        one_container['weather_score'] = one_container.apply(
            lambda r: weather_risk_score(
                r['wave_m'], r['wind_kph'], r['storm']
            )[0], axis=1
        )
        avg_weather       = one_container['weather_score'].mean()
        high_weather_days = len(
            one_container[one_container['weather_score'] > 50]
        )
        if   avg_weather >= 50: weather_level = "🔴 Critical"
        elif avg_weather >= 35: weather_level = "🟠 High"
        elif avg_weather >= 20: weather_level = "🟡 Medium"
        else:                   weather_level = "🟢 Low"
        weather_flag = avg_weather >= 20

        # Condition 3 — Port
        container_ports = one_container['final_port'].unique()
        port_risks      = [
            port_risk_dict.get(p, 0) for p in container_ports
        ]
        max_port_risk   = max(port_risks) if port_risks else 0
        riskiest_port   = (
            container_ports[np.argmax(port_risks)]
            if len(port_risks) > 0 else "Unknown"
        )
        if   max_port_risk >= 75: port_level = "🔴 Critical"
        elif max_port_risk >= 50: port_level = "🟠 High"
        elif max_port_risk >= 25: port_level = "🟡 Medium"
        else:                     port_level = "🟢 Low"
        port_flag = max_port_risk >= 25

        # Combined
        flags = sum([shock_flag, weather_flag, port_flag])
        if   flags == 3: combined = "🔴 CRITICAL"
        elif flags == 2: combined = "🟠 HIGH"
        elif flags == 1: combined = "🟡 MEDIUM"
        else:            combined = "🟢 NORMAL"

        fleet_results.append({
            'containerID'      : container_id,
            'combined_alert'   : combined,
            'flags_triggered'  : flags,
            'shock_level'      : shock_level,
            'shock_anomalies'  : shock_count,
            'max_shock_g'      : round(max_shock, 3),
            'shock_flag'       : shock_flag,
            'weather_level'    : weather_level,
            'avg_weather_risk' : round(avg_weather, 1),
            'high_weather_days': high_weather_days,
            'weather_flag'     : weather_flag,
            'port_level'       : port_level,
            'riskiest_port'    : riskiest_port,
            'max_port_risk'    : round(max_port_risk, 1),
            'port_flag'        : port_flag,
            'real_damages'     : int(
                shock_hist['damaged'].sum()
            ),
            'cargo_type'       : one_container['type'].mode()[0],
        })

    return pd.DataFrame(fleet_results), port_scores