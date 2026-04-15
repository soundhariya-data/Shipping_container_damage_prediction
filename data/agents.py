"""
agents.py — 4 LangChain Agents for Container Risk Analysis
Fixed version — correct ReAct prompt format
"""

import pandas as pd
import numpy as np
from prophet_analysis import (
    run_prophet,
    weather_risk_score,
    port_risk_score,
)
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ═══════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════
_df          = None
_port_scores = None
_llm         = None


def _get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model          = "gemini-2.5-flash",
        google_api_key = api_key,
        temperature    = 0.3,
        convert_system_message_to_human = True
    )


# ═══════════════════════════════════════════════════════════════
# AGENT 1 TOOLS — Sensor Monitoring
# ═══════════════════════════════════════════════════════════════

@tool
def get_shock_anomalies(container_id: str) -> str:
    """Runs Prophet on a container and returns shock_g anomaly details. Input: container ID string e.g. S1F0EGMT"""
    try:
        shock_hist, _, _, _, _ = run_prophet(_df, container_id, forecast_days=0)
        anomalies = shock_hist[shock_hist['anomaly'] == True]

        if len(anomalies) == 0:
            return f"Container {container_id}: NO anomalies. All {len(shock_hist)} days normal."

        result = (
            f"Container {container_id} SHOCK REPORT:\n"
            f"Days: {len(shock_hist)} | Anomalies: {len(anomalies)} | "
            f"Real damages: {int(shock_hist['damaged'].sum())} | "
            f"Max shock: {shock_hist['actual'].max():.3f}g\n"
            f"Normal range: {shock_hist['yhat_lower'].mean():.2f}g to {shock_hist['yhat_upper'].mean():.2f}g\n"
            f"Top anomalies:\n"
        )
        for _, row in anomalies.nlargest(5, 'actual').iterrows():
            result += f"  {str(row['ds'])[:10]}: actual={row['actual']:.3f}g expected={row['yhat']:.3f}g damaged={int(row['damaged'])}\n"
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_sensor_stats(container_id: str) -> str:
    """Returns sensor statistics for a container. Input: container ID string e.g. S1F0EGMT"""
    try:
        one = _df[_df['containerID'] == container_id].copy()
        if len(one) == 0:
            return f"Container {container_id} not found."
        return (
            f"Container {container_id} SENSOR STATS:\n"
            f"Records: {len(one)} | Cargo: {one['type'].mode()[0]}\n"
            f"Shock avg/max/min: {one['shock_g'].mean():.3f}g / {one['shock_g'].max():.3f}g / {one['shock_g'].min():.3f}g\n"
            f"Vibration avg/max: {one['vibration_hz'].mean():.2f}Hz / {one['vibration_hz'].max():.2f}Hz\n"
            f"Temperature avg/max: {one['temperature_c'].mean():.2f}C / {one['temperature_c'].max():.2f}C\n"
            f"Humidity avg/max: {one['humidity_pct'].mean():.1f}% / {one['humidity_pct'].max():.1f}%\n"
        )
    except Exception as e:
        return f"Error: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# AGENT 2 TOOLS — Weather Risk
# ═══════════════════════════════════════════════════════════════

@tool
def get_weather_risk(container_id: str) -> str:
    """Returns weather risk score based on wave height, wind and storm levels. Input: container ID string e.g. S1F0EGMT"""
    try:
        one = _df[_df['containerID'] == container_id].copy()
        if len(one) == 0:
            return f"Container {container_id} not found."
        scores = one.apply(lambda r: weather_risk_score(r['wave_m'], r['wind_kph'], r['storm'])[0], axis=1)
        return (
            f"Container {container_id} WEATHER RISK:\n"
            f"Avg risk: {scores.mean():.1f}/100 | Peak: {scores.max():.1f}/100\n"
            f"High risk days>50: {int((scores > 50).sum())} | Critical days>75: {int((scores > 75).sum())}\n"
            f"Wave: avg={one['wave_m'].mean():.2f}m max={one['wave_m'].max():.2f}m days>3m={int((one['wave_m'] > 3).sum())}\n"
            f"Wind: avg={one['wind_kph'].mean():.1f}kph max={one['wind_kph'].max():.1f}kph\n"
            f"Storms: {one['storm'].value_counts().to_dict()}\n"
        )
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_storm_events(container_id: str) -> str:
    """Returns severe storm events Orange/Red for a container. Input: container ID string e.g. S1F0EGMT"""
    try:
        one = _df[_df['containerID'] == container_id].copy()
        one = one.reset_index()
        if 'timestamp' not in one.columns:
            one = one.rename(columns={'index': 'timestamp'})
        storms = one[one['storm'].isin(['Orange', 'Red', '1'])]
        if len(storms) == 0:
            return f"Container {container_id}: No severe storm events found."
        result = f"Container {container_id} STORMS: total={len(storms)} with_damage={int(storms['damaged(T)'].sum())}\n"
        for _, row in storms.head(8).iterrows():
            result += f"  {str(row['timestamp'])[:10]}: storm={row['storm']} wave={row['wave_m']}m wind={row['wind_kph']}kph shock={row['shock_g']:.2f}g damaged={int(row['damaged(T)'])}\n"
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# AGENT 3 TOOLS — Port Risk
# ═══════════════════════════════════════════════════════════════

@tool
def get_port_risk(container_id: str) -> str:
    """Returns port risk scores for all ports visited by a container. Input: container ID string e.g. S1F0EGMT"""
    try:
        one = _df[_df['containerID'] == container_id].copy()
        if len(one) == 0:
            return f"Container {container_id} not found."
        risk_dict  = dict(zip(_port_scores['final_port'], _port_scores['risk_score']))
        label_dict = dict(zip(_port_scores['final_port'], _port_scores['risk_label']))
        drate_dict = dict(zip(_port_scores['final_port'], _port_scores['damage_rate']))
        port_data  = sorted([
            (p, risk_dict.get(p,0), label_dict.get(p,'?'), drate_dict.get(p,0), int((one['final_port']==p).sum()))
            for p in one['final_port'].unique()
        ], key=lambda x: x[1], reverse=True)
        result = f"Container {container_id} PORT RISK: {len(port_data)} ports\n"
        for port, score, label, drate, visits in port_data:
            result += f"  {label} {port}: score={score}/100 damage_rate={drate:.1f}% visits={visits}\n"
        if port_data:
            result += f"RISKIEST: {port_data[0][0]} score={port_data[0][1]}/100\n"
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_damage_by_port(container_id: str) -> str:
    """Returns damage events grouped by port and route. Input: container ID string e.g. S1F0EGMT"""
    try:
        one   = _df[_df['containerID'] == container_id].copy()
        total = int(one['damaged(T)'].sum())
        if total == 0:
            return f"Container {container_id}: No damage events recorded."
        port_dmg = one.groupby('final_port').agg(visits=('damaged(T)','count'), damaged=('damaged(T)','sum')).reset_index()
        port_dmg = port_dmg[port_dmg['damaged'] > 0]
        port_dmg['rate'] = (port_dmg['damaged'] / port_dmg['visits'] * 100).round(1)
        route_dmg = one.groupby('route').agg(damaged=('damaged(T)','sum')).reset_index()
        route_dmg = route_dmg[route_dmg['damaged'] > 0].sort_values('damaged', ascending=False)
        result = f"Container {container_id} DAMAGE: total={total}\nPorts:\n"
        for _, row in port_dmg.iterrows():
            result += f"  {row['final_port']}: {int(row['damaged'])} events ({row['rate']}% rate)\n"
        result += "Routes:\n"
        for _, row in route_dmg.iterrows():
            result += f"  {row['route']}: {int(row['damaged'])} events\n"
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# AGENT BUILDER — generic ReAct agent
# ═══════════════════════════════════════════════════════════════

def _build_agent(tools, role, max_iterations=5):
    prompt = PromptTemplate.from_template(
        "You are " + role + "\n\n"
        "Tools:\n{tools}\n\n"
        "Format:\n"
        "Question: {input}\n"
        "Thought: what do I need\n"
        "Action: tool from [{tool_names}]\n"
        "Action Input: container_id string only\n"
        "Observation: result\n"
        "...repeat if needed...\n"
        "Thought: I have enough info\n"
        "Final Answer: complete analysis\n\n"
        "Begin!\n"
        "Question: {input}\n"
        "Thought:{agent_scratchpad}"
    )
    agent = create_react_agent(_llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=False
    )


def build_sensor_agent():
    return _build_agent(
        [get_shock_anomalies, get_sensor_stats],
        "a Sensor Monitoring Agent analyzing container shock anomalies."
    )

def build_weather_agent():
    return _build_agent(
        [get_weather_risk, get_storm_events],
        "a Weather Risk Agent analyzing sea conditions and storm impact."
    )

def build_port_agent():
    return _build_agent(
        [get_port_risk, get_damage_by_port],
        "a Port Risk Agent analyzing historical port damage patterns."
    )


# ═══════════════════════════════════════════════════════════════
# AGENT 4 — Decision Agent
# ═══════════════════════════════════════════════════════════════

def build_decision_agent():
    sensor_agent  = build_sensor_agent()
    weather_agent = build_weather_agent()
    port_agent    = build_port_agent()

    @tool
    def ask_sensor_agent(container_id: str) -> str:
        """Consults Sensor Agent for shock anomaly analysis. Input: container ID string e.g. S1F0EGMT"""
        try:
            out = sensor_agent.invoke({"input": f"Analyze shock anomalies for container {container_id}."})
            return out.get("output", "No output")
        except Exception as e:
            return f"Sensor error: {str(e)}"

    @tool
    def ask_weather_agent(container_id: str) -> str:
        """Consults Weather Agent for sea condition analysis. Input: container ID string e.g. S1F0EGMT"""
        try:
            out = weather_agent.invoke({"input": f"Evaluate weather risks for container {container_id}."})
            return out.get("output", "No output")
        except Exception as e:
            return f"Weather error: {str(e)}"

    @tool
    def ask_port_agent(container_id: str) -> str:
        """Consults Port Agent for port risk analysis. Input: container ID string e.g. S1F0EGMT"""
        try:
            out = port_agent.invoke({"input": f"Identify port risks for container {container_id}."})
            return out.get("output", "No output")
        except Exception as e:
            return f"Port error: {str(e)}"

    decision_prompt = PromptTemplate.from_template(
        "You are the Decision Agent — master coordinator for container risk.\n"
        "Consult ALL 3 agents then give recommendations.\n\n"
        "Tools:\n{tools}\n\n"
        "Format:\n"
        "Question: {input}\n"
        "Thought: consult sensor agent first\n"
        "Action: ask_sensor_agent\n"
        "Action Input: container_id\n"
        "Observation: sensor findings\n"
        "Thought: consult weather agent\n"
        "Action: ask_weather_agent\n"
        "Action Input: container_id\n"
        "Observation: weather findings\n"
        "Thought: consult port agent\n"
        "Action: ask_port_agent\n"
        "Action Input: container_id\n"
        "Observation: port findings\n"
        "Thought: combine all findings\n"
        "Final Answer:\n"
        "CONTAINER STATUS: Container [ID] flagged due to...\n"
        "ROOT CAUSE: ...\n"
        "RISK LEVEL: Low/Medium/High/Critical\n"
        "RECOMMENDATIONS:\n"
        "1. [finding + cargo] -> [action]\n"
        "2. [port finding] -> [action]\n"
        "3. [forecast] -> [action]\n\n"
        "Tool names: [{tool_names}]\n\n"
        "Begin!\n"
        "Question: {input}\n"
        "Thought:{agent_scratchpad}"
    )
    agent = create_react_agent(_llm, [ask_sensor_agent, ask_weather_agent, ask_port_agent], decision_prompt)
    return AgentExecutor(
        agent=agent,
        tools=[ask_sensor_agent, ask_weather_agent, ask_port_agent],
        verbose=True, max_iterations=8,
        handle_parsing_errors=True,
        return_intermediate_steps=False
    )


# ═══════════════════════════════════════════════════════════════
# SETUP + RUN
# ═══════════════════════════════════════════════════════════════

def setup(df, api_key):
    global _df, _port_scores, _llm
    _df          = df
    _port_scores = port_risk_score(df)
    _llm         = _get_llm(api_key)


def run_agent_analysis(container_id, df, api_key):
    setup(df, api_key)
    results = {}
    for name, builder, q in [
        ("sensor",   build_sensor_agent,   f"Analyze shock anomalies for container {container_id}."),
        ("weather",  build_weather_agent,  f"Evaluate weather risks for container {container_id}."),
        ("port",     build_port_agent,     f"Identify port risks for container {container_id}."),
        ("decision", build_decision_agent, f"Full risk analysis and recommendations for container {container_id}."),
    ]:
        print(f"\n[{name.upper()} AGENT] Running...")
        try:
            out = builder().invoke({"input": q})
            results[name] = out.get("output", "No output")
        except Exception as e:
            results[name] = f"Error: {str(e)}"
    return results