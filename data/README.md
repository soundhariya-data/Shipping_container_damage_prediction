```

---

## What Each Function Does
```
Function 1 → run_prophet_for_variable()
             Runs Prophet for ANY column
             shock_g OR wave_m

Function 2 → run_prophet()
             Calls Function 1 TWICE
             Once for shock_g, once for wave_m

Function 3 → weather_risk_score()
             Scores weather danger 0-100

Function 4 → port_risk_score()
             Scores every port by damage history

#Function 5 → get_future_date_risk()
             User picks a future date
             Returns predicted shock + wave + port risk

#Function 6 → fleet_wide_analysis()
             #Runs everything for all 10 containers


prophet_analysis.py
│
├── PORT_COORDINATES (dictionary)
│   └── lat/lon for every port
│       Used by: app.py → draw_india_map()
│
├── run_prophet_for_variable(one_container, variable, forecast_days)
│   └── Called by: run_prophet()
│       What it does:
│       → Takes one container's data
│       → Renames columns to ds, y (Prophet format)
│       → Creates Prophet model
│       → Trains model on historical data
│       → Predicts future values
│       → Flags anomalies (actual > yhat_upper)
│       Returns: historical df, future df, model
│
├── run_prophet(df, container_id, forecast_days)
│   └── Called by: app.py (Single Container page)
│                  agents.py → get_shock_anomalies()
│                  fleet_wide_analysis()
│       What it does:
│       → Filters one container from full df
│       → Calls run_prophet_for_variable() TWICE
│           once for shock_g
│           once for wave_m
│       → Adds metadata back (containerID, damaged, route etc.)
│       Returns: shock_hist, shock_future,
│                wave_hist, wave_future,
│                one_container
│
├── weather_risk_score(wave_m, wind_kph, storm)
│   └── Called by: app.py (Single Container page)
│                  agents.py → get_weather_risk()
│                  fleet_wide_analysis()
│       What it does:
│       → Scores wave height    (0-40 pts)
│       → Scores wind speed     (0-30 pts)
│       → Scores storm level    (0-30 pts)
│       → Adds them up → total score 0-100
│       → Returns label (🟢/🟡/🟠/🔴)
│       Returns: (score, label)
│
├── port_risk_score(df)
│   └── Called by: app.py (Single Container + Fleet pages)
│                  agents.py → get_port_risk()
│                  fleet_wide_analysis()
│       What it does:
│       → Groups all data by final_port
│       → Counts total visits + damage events
│       → Calculates damage_rate = damaged/total × 100
│       → Calculates risk_score  = damage_rate × 3
│       → Adds lat/lon coordinates for map
│       Returns: port_scores dataframe
│
├── get_future_date_risk(shock_future, wave_future,
│                        port_scores, selected_date,
│                        container_ports)
│   └── Called by: app.py (Single Container → date picker)
│       What it does:
│       → Finds the row for selected future date
│       → Gets predicted shock_g for that date
│       → Gets predicted wave_m for that date
│       → Calculates weather risk score
│       → Finds port risk for container's ports
│       → Checks 3 flags:
│           shock_flag   → is shock rising toward upper bound?
│           weather_flag → is weather score > 25?
│           port_flag    → is port risk > 25?
│       → Combines flags → overall risk level
│       Returns: dict with all risk info
│
├── parse_route_ports(route_str)
│   └── Called by: get_route_map_data()
│       What it does:
│       → Splits "Mumbai-Kochi-Colombo"
│         into ["Mumbai", "Kochi", "Colombo"]
│       Returns: list of port names
│
├── get_route_map_data(df, container_id=None)
│   └── Called by: app.py (Route Map + Fleet pages)
│       What it does:
│       → Gets all unique routes
│       → Calls parse_route_ports() for each route
│       → Looks up coordinates for each port
│       → Creates segments (port1→port2) with:
│           lat/lon of both ports
│           list of containers on that route
│           damage count on that route
│       Returns: segments dataframe for map drawing
│
└── fleet_wide_analysis(df, forecast_days)
    └── Called by: app.py (Fleet Dashboard page)
        What it does:
        → Loops through ALL 10 containers
        → For each container:
            calls run_prophet()
            calculates weather scores
            calculates port risks
            checks 3 conditions
            assigns combined alert level
        Returns: fleet_df, port_scores




agents.py FILE STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

GLOBAL STATE
    ↓
_df (pandas df) ←─────────── setup(df, api_key)
_port_scores ←────────────── port_risk_score(df)  
_llm (Gemini) ←───────────── _get_llm(api_key)

AGENT 1 TOOLS ─ Sensor Monitoring
    ↓
get_shock_anomalies() ←─────── run_prophet(_df, container_id)
    ↓                           ↓
    └── shock_hist[anomaly=True] → Top 5 anomalies report
get_sensor_stats() ←────────── _df[container_id] stats (shock/vib/temp/hum)

AGENT 2 TOOLS ─ Weather Risk
    ↓
get_weather_risk() ←────────── weather_risk_score(wave_m, wind_kph, storm)
    ↓                           ↓
    └── scores.mean()/max() → High/critical day counts + wave/wind/storm stats
get_storm_events() ←────────── _df[storm in ['Orange','Red','1']] top 8

AGENT 3 TOOLS ─ Port Risk
    ↓
get_port_risk() ←───────────── _port_scores dict × _df[container_id].unique ports
    ↓                           ↓
    └── sorted by risk_score → Label + score + damage_rate + visits
get_damage_by_port() ←──────── groupby('final_port') + groupby('route')

AGENT BUILDERS ─ ReAct Agents (max_iterations=5)
    ↓
build_sensor_agent() → create_react_agent(llm, [shock,stats], sensor_prompt)
build_weather_agent() → create_react_agent(llm, [weather,storms], weather_prompt)  
build_port_agent() → create_react_agent(llm, [port,damage], port_prompt)

DECISION AGENT (max_iterations=8)
    ↓
build_decision_agent()
    ↓
    ├── ask_sensor_agent() → sensor_agent.invoke()
    ├── ask_weather_agent() → weather_agent.invoke()
    └── ask_port_agent() → port_agent.invoke()
    ↓
    └── create_react_agent(llm, [ask_* tools], decision_prompt)
        ↓
        └── ALWAYS consults: sensor → weather → port → Final Answer

RUNNER
    ↓
run_agent_analysis(container_id, df, api_key)
    ↓
    ├── setup(df, api_key)
    ├── sensor_agent.invoke() → results['sensor']
    ├── weather_agent.invoke() → results['weather']
    ├── port_agent.invoke() → results['port']  
    └── decision_agent.invoke() → results['decision']

═══════════════════════════════════════════════════════════════════════════════
INPUT: container_id="S1F0EGMT" (string only)
OUTPUT: dict{'sensor':str, 'weather':str, 'port':str, 'decision':str}
DF REQUIRED COLUMNS: containerID, shock_g, vibration_hz, temperature_c, humidity_pct, 
                     wave_m, wind_kph, storm, final_port, damaged(T), damaged, route