import re

import subprocess

import time



import pandas as pd

import streamlit as st



from src.services.investigation_service import (

    InvestigationError,

    list_experiments,

    run_investigation

)



from src.services.system_monitor import (

    get_system_status,

    get_model_recommendations,

    get_ollama_info

)



from src.agents.llm_client import (

    get_active_model,

    set_model

)





# ==============================================================================

# PERFORMANCE-OPTIMIZED CACHING LAYER

# ==============================================================================



@st.cache_data(ttl=60, show_spinner=False)

def cached_experiments():

    """Cache experiment list for 60s to prevent disk scanning on page switch."""

    return list_experiments()





@st.cache_data(ttl=8, show_spinner=False)

def cached_system_status():

    """Cache telemetry so reruns and monitor fragments reuse recent readings."""

    return get_system_status()





@st.cache_data(ttl=180, show_spinner=False)

def cached_model_recommendations():

    """Cache RAM-based model recommendations for 3 minutes."""

    return get_model_recommendations()





@st.cache_data(ttl=10, show_spinner=False)

def cached_active_model():

    """Avoid redundant LLM queries during a single rerun."""

    try:

        return get_active_model()

    except Exception:

        return "Unknown"





# ==============================================================================

# PAGE CONFIGURATION & STYLING

# ==============================================================================



st.set_page_config(

    page_title="MIRA | Intelligence Console",

    page_icon="◈",

    layout="wide",

    initial_sidebar_state="expanded"

)



st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Inter:wght@300;400;500;600;700;800&display=swap');



:root {

    --bg-base: #070a11;

    --bg-surface: #0d131f;

    --bg-elevated: #131b2c;

    --border-subtle: #1c273c;

    --border-bright: #2d3f5e;

    --cyan: #38bdf8;

    --cyan-glow: rgba(56, 189, 248, 0.15);

    --violet: #a78bfa;

    --emerald: #34d399;

    --rose: #fb7185;

    --amber: #fbbf24;

    --text-primary: #f1f5f9;

    --text-secondary: #94a3b8;

    --text-muted: #64748b;

}



/* Background & Typography */

.stApp {

    background: radial-gradient(circle at 15% 10%, #0d182b 0%, var(--bg-base) 65%);

    color: var(--text-primary);

    font-family: 'Inter', -apple-system, sans-serif;

}



[data-testid="stSidebar"] {

    background: #090e18;

    border-right: 1px solid var(--border-subtle);

}



[data-testid="stSidebar"] > div:first-child {

    padding-top: 1.2rem;

}



h1, h2, h3, h4 {

    color: var(--text-primary) !important;

    font-weight: 700 !important;

    letter-spacing: -0.02em;

}



.mono {

    font-family: 'DM Mono', monospace !important;

}



/* Brand Identity */

.mira-brand-box {

    display: flex;

    align-items: center;

    gap: 12px;

    padding: 6px 4px 10px 4px;

}



.mira-brand-icon {

    font-size: 26px;

    color: var(--cyan);

    text-shadow: 0 0 16px rgba(56, 189, 248, 0.6);

}



.mira-brand {

    font-size: 24px;

    font-weight: 800;

    letter-spacing: 4px;

    background: linear-gradient(135deg, #38bdf8 0%, #a78bfa 100%);

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

}



.mira-subtitle {

    color: var(--text-muted);

    font-size: 10px;

    letter-spacing: 2.5px;

    text-transform: uppercase;

    font-weight: 600;

}



/* Section Header Labels */

.section-label {

    color: var(--cyan);

    font-family: 'DM Mono', monospace;

    font-size: 10px;

    font-weight: 600;

    letter-spacing: 2px;

    text-transform: uppercase;

    margin-bottom: 8px;

    display: flex;

    align-items: center;

    gap: 6px;

}



.section-label::before {

    content: "";

    display: inline-block;

    width: 6px;

    height: 6px;

    background: var(--cyan);

    border-radius: 50%;

    box-shadow: 0 0 8px var(--cyan);

}



/* Modern Card Panels */

.mira-card {

    background: linear-gradient(160deg, #101828, #0b111c);

    border: 1px solid var(--border-subtle);

    border-radius: 12px;

    padding: 18px 20px;

    margin-bottom: 14px;

    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);

    transition: border-color 0.2s ease;

}



.mira-card:hover {

    border-color: var(--border-bright);

}



/* Native Metric Override */

div[data-testid="stMetric"] {

    background: linear-gradient(145deg, #111929, #0d1320);

    border: 1px solid var(--border-subtle);

    padding: 14px 16px;

    border-radius: 10px;

    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);

}



div[data-testid="stMetric"] label {

    font-size: 11px !important;

    text-transform: uppercase;

    letter-spacing: 1px;

    color: var(--text-secondary) !important;

}



div[data-testid="stMetricValue"] {

    color: var(--cyan) !important;

    font-family: 'DM Mono', monospace !important;

    font-size: 24px !important;

}



/* Status Badges */

.status-pill {

    display: inline-flex;

    align-items: center;

    gap: 6px;

    padding: 4px 12px;

    border-radius: 9999px;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1px;

    text-transform: uppercase;

    font-family: 'DM Mono', monospace;

}



.status-healthy {

    background: rgba(52, 211, 153, 0.12);

    color: #34d399;

    border: 1px solid rgba(52, 211, 153, 0.35);

}



.status-critical {

    background: rgba(251, 113, 133, 0.12);

    color: #fb7185;

    border: 1px solid rgba(251, 113, 133, 0.35);

}



.status-warning {

    background: rgba(251, 191, 36, 0.12);

    color: #fbbf24;

    border: 1px solid rgba(251, 191, 36, 0.35);

}



/* Buttons */

.stButton > button[kind="primary"] {

    background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%) !important;

    border: 1px solid rgba(255, 255, 255, 0.15) !important;

    color: #ffffff !important;

    font-weight: 700 !important;

    border-radius: 8px !important;

    min-height: 44px !important;

    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);

    transition: all 0.2s ease;

}



.stButton > button[kind="primary"]:hover {

    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);

    transform: translateY(-1px);

}



.stButton > button {

    border-radius: 8px;

    border: 1px solid var(--border-subtle);

    background: #121927;

    color: var(--text-primary);

    transition: all 0.15s ease;

}



.stButton > button:hover {

    border-color: var(--cyan);

    color: var(--cyan);

}



/* Progress Bars */

.stProgress > div > div > div > div {

    background: linear-gradient(90deg, #0284c7, #38bdf8) !important;

    border-radius: 4px;

}



/* Tabs */

div[data-testid="stTabs"] button {

    font-size: 13px !important;

    font-weight: 600 !important;

    color: var(--text-muted) !important;

    padding: 10px 18px !important;

}



div[data-testid="stTabs"] button[aria-selected="true"] {

    color: var(--cyan) !important;

    border-bottom: 2px solid var(--cyan) !important;

    background: rgba(56, 189, 248, 0.05) !important;

}



/* Expanders */

[data-testid="stExpander"] {

    border: 1px solid var(--border-subtle) !important;

    border-radius: 8px !important;

    background: #0d1421 !important;

}



/* Live Pulse Indicators */

.pulse-dot {

    display: inline-block;

    width: 8px;

    height: 8px;

    border-radius: 50%;

    margin-right: 6px;

}

.pulse-green {

    background-color: #34d399;

    box-shadow: 0 0 8px #34d399;

}

.pulse-red {

    background-color: #fb7185;

    box-shadow: 0 0 8px #fb7185;

}



/* Seamless fragment rendering */

div[data-stale="true"] {

    opacity: 1 !important;

    filter: none !important;

}

</style>

""", unsafe_allow_html=True)





# ==============================================================================

# DISPLAY & HELPER FUNCTIONS

# ==============================================================================



def display_metrics(metrics):

    """Render 4-column performance metrics grid with fallback handling."""

    metric_names = [

        ("accuracy", "Accuracy"),

        ("precision", "Precision"),

        ("recall", "Recall"),

        ("f1_score", "F1 Score")

    ]



    columns = st.columns(4)



    for column, (key, label) in zip(columns, metric_names):

        value = (metrics or {}).get(key)

        if value is not None:

            column.metric(label, f"{value:.4f}")

        else:

            column.metric(label, "N/A")





def display_section(title, content):

    """Render a structured sub-section with JSON/text formatting."""

    st.markdown(f"#### {title}")



    if isinstance(content, (dict, list)):

        st.json(content)

    elif content is not None:

        st.write(content)

    else:

        st.info("No information available.")





@st.fragment

def render_investigation_control(selected_id):

    """Investigation execution controller with multi-stage progress tracking."""

    active_model = cached_active_model()

    st.caption(f"Active LLM: **{active_model}**")



    if st.button(

        "▶  INITIATE INVESTIGATION",

        type="primary",

        use_container_width=True,

        key="initiate_investigation"

    ):

        stage_labels = {

            "planning": "Selecting diagnostic tools",

            "execute": "Executing diagnostic tools",

            "investigate": "Investigating experiment",

            "verify": "Verifying evidence",

            "reason": "Running local LLM reasoning",

            "report": "Generating final report",

            "error": "Investigation failed"

        }



        stage_order = [

            "planning",

            "execute",

            "investigate",

            "verify",

            "reason",

            "report"

        ]



        with st.status(

            "MIRA investigation running...",

            expanded=True

        ) as investigation_status:



            progress_bar = st.progress(0)

            current_stage = st.empty()

            completed_stages = []



            def update_progress(node_name):

                if node_name in stage_order:

                    if node_name not in completed_stages:

                        completed_stages.append(node_name)



                    current_stage.markdown(

                        f"◈ **{stage_labels[node_name]}**"

                    )



                    progress_bar.progress(

                        len(completed_stages) / len(stage_order)

                    )



                elif node_name == "error":

                    current_stage.error(stage_labels[node_name])



            try:

                result = run_investigation(

                    selected_id,

                    progress_callback=update_progress

                )



                st.session_state["investigation_result"] = result

                st.session_state["investigated_id"] = selected_id



                history = st.session_state.setdefault(

                    "investigation_history", []

                )



                history.append({

                    "experiment_id": selected_id,

                    "status": result.get("status", "UNKNOWN"),

                    "formatted_report": result.get("formatted_report", ""),

                    "timestamp": time.strftime("%H:%M:%S")

                })



                investigation_status.update(

                    label="Investigation completed successfully",

                    state="complete"

                )



                progress_bar.progress(1.0)

                st.rerun(scope="app")



            except (ValueError, InvestigationError) as error:

                investigation_status.update(

                    label="Investigation failed",

                    state="error"

                )

                st.error(str(error))



            except Exception as error:

                investigation_status.update(

                    label="Unexpected error",

                    state="error"

                )

                st.error(str(error))





# ==============================================================================

# SYSTEM MONITOR (FAST FRAGMENT USING CACHED TELEMETRY)

# ==============================================================================



@st.fragment(run_every="8s")

def render_system_monitor():

    """Live telemetry widget consuming cached_system_status() without queue locking."""

    try:

        # Uses cached reading: instant in memory, no redundant WMI / subprocess thrashing

        status = cached_system_status()



        if not status:

            st.caption("Telemetry unavailable")

            return



        specs = status.get("specs", {})

        load = status.get("load", {})



        col_title, col_btn = st.columns([3, 1])

        with col_title:

            st.markdown(

                '<div class="section-label">SYSTEM MONITOR</div>',

                unsafe_allow_html=True

            )

        with col_btn:

            if st.button("↻", key="refresh_telemetry_btn", help="Refresh hardware telemetry now"):

                cached_system_status.clear()

                st.rerun()



        processor = specs.get("processor", "Local CPU")

        st.caption(f"{processor[:30]}..." if len(processor) > 30 else processor)



        cpu_val = load.get("cpu_percent", 0.0)

        st.progress(

            min(max(cpu_val / 100.0, 0.0), 1.0),

            text=f"CPU {cpu_val:.1f}%"

        )



        ram_used = load.get("ram_used_gb", 0.0)

        ram_total = specs.get("total_ram_gb", 1.0)

        ram_pct = load.get("ram_percent", (ram_used / ram_total * 100 if ram_total else 0.0))

        st.progress(

            min(max(ram_pct / 100.0, 0.0), 1.0),

            text=f"RAM {ram_used:.1f} / {ram_total:.1f} GB"

        )



        gpu = load.get("gpu")

        if gpu:

            st.caption(f"GPU: {gpu.get('name', 'NVIDIA GPU')}")

            

            # Dedicated VRAM allocation bar (true activity indicator for LLMs)

            mem_used = gpu.get("memory_used_mb", 0.0)

            mem_total = gpu.get("memory_total_mb", 1.0)

            vram_pct = gpu.get("vram_percent", (mem_used / mem_total * 100.0) if mem_total else 0.0)

            

            st.progress(

                min(max(vram_pct / 100.0, 0.0), 1.0),

                text=f"VRAM {vram_pct:.1f}% ({mem_used:.0f}/{mem_total:.0f} MB)"

            )



            # Compute Core & Temperature tracking

            gpu_util = gpu.get("utilization", 0.0)

            compute_label = f"Compute {gpu_util:.1f}%" if gpu_util > 0 else ("Compute Idle" if vram_pct > 0 else "Compute 0.0%")

            st.progress(

                min(max(gpu_util / 100.0, 0.0), 1.0),

                text=f"{compute_label} · {gpu.get('temperature', 0):.0f}°C"

            )

        else:

            st.caption("GPU telemetry unavailable")



        cores = specs.get("cpu_cores", "?")

        threads = specs.get("cpu_threads", "?")

        st.caption(f"{cores} cores / {threads} threads")



        ollama = load.get("ollama", {})

        if ollama.get("connected", False):

            st.markdown(

                '<span class="pulse-dot pulse-green"></span>'

                '<span style="color:#34d399;font-size:12px;font-weight:600;">Ollama Connected</span>',

                unsafe_allow_html=True

            )

        else:

            st.markdown(

                '<span class="pulse-dot pulse-red"></span>'

                '<span style="color:#fb7185;font-size:12px;font-weight:600;">Ollama Disconnected</span>',

                unsafe_allow_html=True

            )



    except Exception as error:

        st.warning(f"System monitor unavailable: {error}")





def install_model(model_name):

    """Pull Ollama model via CLI with live stream and security validation."""

    if not re.fullmatch(r"[A-Za-z0-9._:/-]+", model_name):

        st.error("Invalid model name.")

        return



    try:

        process = subprocess.Popen(

            ["ollama", "pull", model_name],

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            encoding="utf-8",

            errors="replace"

        )



        output_area = st.empty()

        output_lines = []



        with st.status(

            f"Downloading {model_name}...",

            expanded=True

        ) as download_status:



            for line in process.stdout:

                output_lines.append(line.rstrip())

                output_area.code("\n".join(output_lines[-12:]))



            return_code = process.wait()



            if return_code == 0:

                download_status.update(

                    label=f"✓ {model_name} installed successfully",

                    state="complete"

                )

                st.success("Model installation completed.")

                cached_system_status.clear()

            else:

                download_status.update(

                    label="Model installation failed",

                    state="error"

                )

                st.error("Ollama could not install this model.")



        if return_code == 0:

            st.rerun()



    except FileNotFoundError:

        st.error("Ollama executable was not found.")

    except Exception as error:

        st.error(f"Installation error: {error}")





# ==============================================================================

# DATA LOAD & STATE INITIALIZATION

# ==============================================================================



try:

    experiments = cached_experiments()

except InvestigationError as error:

    st.error(str(error))

    st.stop()



if not experiments:

    st.warning("No experiments available.")

    st.stop()



experiment_lookup = {

    experiment["experiment_id"]: experiment

    for experiment in experiments

}



try:

    system_status = cached_system_status()

except Exception:

    system_status = None



if system_status and "load" in system_status and "ollama" in system_status["load"]:

    ollama_info = system_status["load"]["ollama"]

else:

    ollama_info = {"connected": False, "models": []}



active_model_cached = cached_active_model()





# ==============================================================================

# SIDEBAR NAVIGATION & CONTROLS

# ==============================================================================



with st.sidebar:
    st.markdown(
        '<div class="mira-brand-box">'
        '<span class="mira-brand-icon">◈</span>'
        '<div>'
        '<div class="mira-brand">MIRA</div>'
        '<div class="mira-subtitle">Intelligence Console</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()



    st.markdown('<div class="section-label">NAVIGATION</div>', unsafe_allow_html=True)

    page = st.radio(

        "Navigation",

        [

            "Overview",

            "Investigation History",

            "System & Models"

        ],

        label_visibility="collapsed"

    )



    st.divider()



    st.markdown('<div class="section-label">EXPERIMENT SELECTOR</div>', unsafe_allow_html=True)

    selected_id = st.selectbox(

        "Experiment",

        options=list(experiment_lookup.keys()),

        format_func=lambda exp_id: f"{exp_id} · {experiment_lookup[exp_id].get('name', 'Run')}",

        label_visibility="collapsed"

    )



    st.divider()



    st.markdown('<div class="section-label">LOCAL MODEL</div>', unsafe_allow_html=True)



    if ollama_info.get("connected", False):

        installed_models = ollama_info.get("models", [])

        model_names = [model["name"] for model in installed_models]



        if model_names:

            if active_model_cached not in model_names:

                model_names.insert(0, active_model_cached)



            selected_model = st.selectbox(

                "Active model",

                options=model_names,

                index=model_names.index(active_model_cached) if active_model_cached in model_names else 0,

                label_visibility="collapsed"

            )



            if selected_model != active_model_cached:

                try:

                    set_model(selected_model)

                    cached_active_model.clear()

                    st.success(f"Active: {selected_model}")

                    st.rerun()

                except Exception as error:

                    st.error(str(error))



            for model in installed_models:

                if model.get("name") == active_model_cached:

                    state_label = "● Loaded in memory" if model.get("loaded") else "○ Installed · not loaded"

                    st.caption(state_label)

                    break

        else:

            st.info("No installed models found.")



        with st.expander("＋ Add model"):

            st.caption("Enter an Ollama model name to install:")

            new_model = st.text_input(

                "Model name",

                placeholder="e.g. llama3.2:3b",

                label_visibility="collapsed"

            )



            if st.button("Install model", use_container_width=True):

                if new_model.strip():

                    install_model(new_model.strip())

                else:

                    st.warning("Enter a model name.")



        with st.expander("Recommended models"):

            try:

                recommendations = cached_model_recommendations()



                for recommendation in recommendations:

                    st.markdown(f"**{recommendation['name']}**")

                    st.caption(recommendation.get("description", ""))



                    if recommendation.get("recommended"):

                        st.caption("✓ Suitable for current available RAM")

                    else:

                        st.caption("! May require additional memory")



                    if recommendation["name"] not in model_names:

                        if st.button(

                            f"Install {recommendation['name']}",

                            key=f"install_{recommendation['name']}"

                        ):

                            install_model(recommendation["name"])

            except Exception as error:

                st.caption(f"Recommendations unavailable: {error}")

    else:

        st.error("Ollama is not connected.")

        st.caption("Start Ollama to use local models.")



    st.divider()



    # Telemetry widget runs seamlessly in the background

    render_system_monitor()





# ==============================================================================

# MAIN VIEWPORT

# ==============================================================================



selected_experiment = experiment_lookup[selected_id]



# Header Bar

head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown('<div class="section-label">ML INVESTIGATION PLATFORM / LOCAL INSTANCE</div>', unsafe_allow_html=True)
    st.title("Intelligence Overview" if page == "Overview" else ("Investigation History" if page == "Investigation History" else "System & Model Information"))
with head_col2:
    st.markdown(
        '<div style="text-align: right; padding-top: 14px;">'
        '<span class="status-pill status-healthy">● System Ready</span>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("Investigate model failures. Trace evidence. Understand root causes.")
st.divider()





# ------------------------------------------------------------------------------

# PAGE 1: OVERVIEW

# ------------------------------------------------------------------------------

if page == "Overview":
    top_columns = st.columns([2, 1])

    with top_columns[0]:
        st.markdown(
            f'<div class="mira-card" style="padding: 16px 20px;">'
            f'<div style="font-size: 11px; color: var(--text-muted); font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;">Selected Experiment</div>'
            f'<div style="font-size: 26px; font-weight: 800; color: var(--cyan); font-family: \'DM Mono\', monospace; margin: 4px 0;">{selected_id}</div>'
            f'<div style="color: var(--text-secondary); font-size: 14px;">{selected_experiment.get("name", "Unnamed Experiment")}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with top_columns[1]:
        train_count = selected_experiment.get('training_records', 0)
        test_count = selected_experiment.get('testing_records', 0)
        st.metric("Training Dataset", f"{train_count:,}")
        st.caption(f"Testing partition: **{test_count:,} records**")



    st.markdown("### Model Performance")

    display_metrics(selected_experiment.get("metrics", {}))



    st.divider()



    left, right = st.columns([1.2, 1])



    with left:

        st.markdown("### Performance Profile")

        metrics_dict = selected_experiment.get("metrics") or {}

        metric_data = {

            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],

            "Score": [

                float(metrics_dict.get("accuracy") or 0.0),

                float(metrics_dict.get("precision") or 0.0),

                float(metrics_dict.get("recall") or 0.0),

                float(metrics_dict.get("f1_score") or 0.0)

            ]

        }

        chart_data = pd.DataFrame(metric_data).set_index("Metric")

        st.bar_chart(chart_data, horizontal=True, height=220)



    with right:

        st.markdown("### Investigation Control")

        st.markdown(

            "Run the MIRA agent to inspect performance, execute "

            "diagnostics, verify evidence and generate a root-cause report."

        )

        render_investigation_control(selected_id)



    # Completed Investigation Inspection

    result = st.session_state.get("investigation_result")

    investigated_id = st.session_state.get("investigated_id")



    if result and investigated_id == selected_id:

        st.divider()

        st.markdown("## Investigation Results")



        status = result.get("status", "UNKNOWN")

        if status == "HEALTHY":

            st.markdown(f'<span class="status-pill status-healthy">● {status}</span>', unsafe_allow_html=True)

        elif status in ["CRITICAL", "HIGH RISK"]:

            st.markdown(f'<span class="status-pill status-critical">● {status}</span>', unsafe_allow_html=True)

        else:

            st.markdown(f'<span class="status-pill status-warning">● {status}</span>', unsafe_allow_html=True)



        tabs = st.tabs([

            "Investigation Report",

            "Diagnostics",

            "Verification",

            "AI Reasoning"

        ])



        with tabs[0]:

            display_metrics(result.get("performance", {}))

            display_section("Dataset Summary", result.get("dataset_summary"))

            display_section("Findings", result.get("findings"))

            display_section("Recommendations", result.get("recommendations"))



            with st.expander("Complete formatted report"):

                st.text(result.get("formatted_report", ""))



        with tabs[1]:

            display_section("Diagnostic Tool Selection", result.get("selection"))

            display_section("Diagnostic Results", result.get("diagnostic_results"))



        with tabs[2]:

            display_section("Evidence Verification", result.get("verification"))



        with tabs[3]:
            st.markdown("### Root-Cause Analysis")
            raw_reasoning = result.get("reasoning", "")
            
            # Format sections into clean visual blocks
            if raw_reasoning:
                # Add markdown headers for standard sections
                formatted_reasoning = (
                    raw_reasoning.replace("ROOT CAUSE", "#### 🔍 Root Cause\n")
                    .replace("EVIDENCE", "\n#### 📊 Evidence\n")
                    .replace("UNCERTAINTIES", "\n#### ❓ Uncertainties\n")
                    .replace("NEXT ACTION", "\n#### 🎯 Next Action\n")
                )
                st.markdown(formatted_reasoning)
            else:
                st.info("No reasoning output available.")



# ------------------------------------------------------------------------------

# PAGE 2: INVESTIGATION HISTORY (UPGRADED)

# ------------------------------------------------------------------------------

elif page == "Investigation History":

    st.markdown("## Investigation History")

    st.caption("Results retained during this Streamlit session.")



    history = st.session_state.get("investigation_history", [])



    if not history:

        st.info("No completed investigations in this session. Head to Overview and click 'Initiate Investigation'.")

    else:

        # Action bar

        act_col1, act_col2 = st.columns([3, 1])

        with act_col1:

            st.caption(f"Total investigations recorded: **{len(history)}**")

        with act_col2:

            if st.button("Clear History", use_container_width=True):

                st.session_state["investigation_history"] = []

                st.session_state["investigation_result"] = None

                st.rerun()



        for idx, item in enumerate(reversed(history)):

            status = item.get("status", "UNKNOWN")

            status_class = "status-healthy" if status == "HEALTHY" else ("status-critical" if status in ["CRITICAL", "HIGH RISK"] else "status-warning")

            time_label = item.get("timestamp", "Recent")



            with st.expander(f"◈ {item.get('experiment_id')}  ·  {status}  ({time_label})", expanded=(idx == 0)):

                st.markdown(f'<span class="status-pill {status_class}">● {status}</span>', unsafe_allow_html=True)

                st.markdown(f"**Experiment:** `{item.get('experiment_id')}`")

                st.text_area(

                    "Formatted Diagnostic Report",

                    value=item.get("formatted_report", "No report available."),

                    height=220,

                    key=f"hist_view_{idx}"

                )





# ------------------------------------------------------------------------------

# PAGE 3: SYSTEM & MODELS

# ------------------------------------------------------------------------------

elif page == "System & Models":
    st.markdown("## System & Model Information")

    try:
        specs = (system_status or {}).get("specs", {})
        load = (system_status or {}).get("load", {})

        st.markdown("### Hardware specifications")
        spec_columns = st.columns(3)
        spec_columns[0].metric("CPU cores", specs.get("cpu_cores", "N/A"))
        spec_columns[1].metric("CPU threads", specs.get("cpu_threads", "N/A"))
        spec_columns[2].metric("Installed RAM", f"{specs.get('total_ram_gb', 0.0):.1f} GB")

        st.markdown(
            f'<div class="mira-card" style="padding: 14px 18px; margin-top: 10px;">'
            f'<div style="font-size: 13px; color: var(--text-secondary); line-height: 1.8;">'
            f'<div><b>Processor:</b> {specs.get("processor", "Unknown")}</div>'
            f'<div><b>Operating system:</b> {specs.get("operating_system", "Unknown")}</div>'
            f'<div><b>Python:</b> {specs.get("python_version", "Unknown")}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Current resource usage")
        load_columns = st.columns(4)
        load_columns[0].metric("CPU", f"{load.get('cpu_percent', 0.0):.1f}%")
        load_columns[1].metric("RAM", f"{load.get('ram_percent', 0.0):.1f}%")

        gpu = load.get("gpu")
        if gpu:
            mem_used = gpu.get("memory_used_mb", 0.0)
            mem_total = gpu.get("memory_total_mb", 1.0)
            vram_pct = gpu.get("vram_percent", (mem_used / mem_total * 100.0) if mem_total else 0.0)

            load_columns[2].metric("VRAM Allocated", f"{vram_pct:.1f}%", f"{mem_used:.0f} / {mem_total:.0f} MB")
            load_columns[3].metric("GPU Compute", f"{gpu.get('utilization', 0.0):.1f}%", f"{gpu.get('temperature', 0):.0f}°C")

            st.caption(
                f"{gpu.get('name', 'NVIDIA GPU')} · "
                f"Power: {gpu.get('power_watts', 0):.1f} W · "
                f"Core: {gpu.get('core_utilization', 0):.1f}% · "
                f"Mem Bus: {gpu.get('memory_bus_utilization', 0):.1f}%"
            )
        else:
            load_columns[2].metric("GPU", "Unavailable")

        st.markdown("### Installed Ollama models")
        if ollama_info.get("connected") and ollama_info.get("models"):
            for model in ollama_info["models"]:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
                        f'<span style="font-size: 16px; font-weight: 700; color: var(--text-primary); font-family: \'DM Mono\', monospace;">'
                        f'{model.get("name")}'
                        f'</span>'
                        f'<span class="status-pill {"status-healthy" if model.get("loaded") else "status-warning"}">'
                        f'{"● Memory Resident" if model.get("loaded") else "○ Installed"}'
                        f'</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    model_columns = st.columns(3)
                    model_columns[0].metric("Disk size", f"{model.get('size_gb', 0.0):.2f} GB")
                    model_columns[1].metric("Parameters", model.get("parameter_size") or "Unknown")
                    model_columns[2].metric("Status", "Loaded" if model.get("loaded") else "Installed")
        else:
            st.info("No installed models detected.")

    except Exception as error:
        st.error(f"Unable to retrieve system information: {error}")





# ==============================================================================

# FOOTER

# ==============================================================================



st.divider()



st.markdown(

    '<div style="text-align:center;color:#64748b;'

    'font-size:11px;letter-spacing:2px;font-family:\'DM Mono\',monospace;">'

    'MIRA · ML INVESTIGATION & ROOT-CAUSE ANALYSIS'

    '<br>LOCAL AI INTELLIGENCE CONSOLE'

    '</div>',

    unsafe_allow_html=True

)