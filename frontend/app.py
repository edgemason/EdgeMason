"""
Streamlit Web Dashboard for EdgeForge / EdgeMason.

Features:
- Drag-and-drop ONNX model upload & calibration data
- Hardware target selection (Raspberry Pi 5/4/3, Jetson Nano, x86_64)
- Hard constraint sliders (Max latency ms, Max memory MB)
- Graph operator analysis & parameter statistics
- Pipeline execution & real-time progress
- Winner showcase & multi-candidate comparison charts
- Direct one-click ZIP package download
"""

import io
import json
import requests
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="EdgeForge — Edge AI Deployment Planner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .winner-card {
        padding: 1.5rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.75rem;
        padding: 0.75rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="main-title">EdgeForge</div>', unsafe_allow_html=True)
st.caption("Hardware-Aware Intelligent Deployment Planner for Edge AI Models")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Target Hardware & Constraints")
    
    api_url = st.text_input("Backend API URL", value="http://127.0.0.1:8000")
    
    hardware_id = st.selectbox(
        "Target Edge Hardware",
        options=["rpi5", "rpi4", "rpi3", "jetson_nano", "x86_64"],
        index=0,
        format_func=lambda x: {
            "rpi5": "🍓 Raspberry Pi 5 (Cortex-A76 @ 2.4GHz)",
            "rpi4": "🍓 Raspberry Pi 4 (Cortex-A72 @ 1.8GHz)",
            "rpi3": "🍓 Raspberry Pi 3 (Cortex-A53 @ 1.4GHz)",
            "jetson_nano": "⚡ NVIDIA Jetson Nano (GPU/ARM)",
            "x86_64": "💻 Standard x86_64 Host / Server",
        }.get(x, x),
    )
    
    priority = st.selectbox(
        "Optimization Priority",
        options=["latency", "memory", "accuracy", "balanced"],
        index=0,
        format_func=lambda x: {
            "latency": "⚡ Latency (Lowest p50 ms)",
            "memory": "💾 Memory (Lowest RAM Delta)",
            "accuracy": "🎯 Accuracy (Highest Top-1 %)",
            "balanced": "⚖️ Balanced (Weighted Multi-Score)",
        }.get(x, x),
    )
    
    execution_provider = st.selectbox(
        "Execution Provider",
        options=["cpu", "auto", "cuda"],
        index=0,
        help="CPU by default. Auto uses the GPU if the target supports it; CUDA requires an NVIDIA GPU."
    )
    
    include_tvm = st.toggle("Include Apache TVM Relay", value=True, help="Compiles TVM Opt-Level-3 candidates alongside ONNX Runtime.")
    
    st.markdown("---")
    st.subheader("🛑 Hard Constraints (Optional)")
    max_latency_ms = st.number_input("Max Latency (ms)", min_value=0.0, value=0.0, step=5.0, help="Leave 0.0 for no constraint.")
    max_memory_mb = st.number_input("Max Memory Delta (MB)", min_value=0.0, value=0.0, step=5.0, help="Leave 0.0 for no constraint.")


# Main Body: Upload & Workflow
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 1. Model & Calibration Upload")
    
    model_file = st.file_uploader(
        "Upload Trained ONNX Model",
        type=["onnx"],
        help="Single-input Float32 ONNX neural network model."
    )
    
    calib_file = st.file_uploader(
        "Upload Calibration / Validation Dataset (Optional)",
        type=["npy", "npz"],
        help="NumPy dataset (.npy for static INT8 calibration, .npz with labels for accuracy testing)."
    )

with col2:
    st.subheader("🔍 2. Quick Graph Inspection")
    if model_file:
        with st.spinner("Inspecting ONNX graph structure..."):
            try:
                analyze_resp = requests.post(
                    f"{api_url}/analyze",
                    files={"file": (model_file.name, model_file.getvalue(), "application/octet-stream")},
                    timeout=30
                )
                if analyze_resp.status_code == 200:
                    analysis_data = analyze_resp.json().get("analysis", {})
                    
                    m_size = analysis_data.get("model_size_mb", 0)
                    m_params = analysis_data.get("total_parameters", 0)
                    m_nodes = analysis_data.get("num_nodes", 0)
                    
                    st.success(f"Loaded: **{model_file.name}**")
                    ic1, ic2, ic3 = st.columns(3)
                    ic1.metric("Model Size", f"{m_size:.2f} MB")
                    ic2.metric("Parameters", f"{m_params:,}")
                    ic3.metric("Nodes", f"{m_nodes:,}")
                    
                    op_types = analysis_data.get("operator_types", {})
                    if op_types:
                        top_ops = pd.DataFrame(
                            list(op_types.items())[:8],
                            columns=["Operator", "Count"]
                        )
                        st.caption("Top Graph Operators")
                        st.bar_chart(top_ops.set_index("Operator"))
                else:
                    st.warning(f"Analysis notice: {analyze_resp.text}")
            except Exception as e:
                st.info(f"Connect to backend at {api_url} to view live graph inspection.")
    else:
        st.info("👈 Please upload an `.onnx` model on the left to begin.")

st.markdown("---")

# Pipeline Execution Button
if st.button("🚀 Run Full Optimization & Benchmark Pipeline", type="primary", use_container_width=True, disabled=not model_file):
    with st.spinner(f"Running pipeline on {hardware_id.upper()} (Quantizing, Compiling TVM/ONNX, Benchmarking & Packaging)..."):
        try:
            files = {"file": (model_file.name, model_file.getvalue(), "application/octet-stream")}
            if calib_file:
                files["calibration_file"] = (calib_file.name, calib_file.getvalue(), "application/octet-stream")
            
            data = {
                "hardware_id": hardware_id,
                "priority": priority,
                "execution_provider": execution_provider,
                "include_tvm": str(include_tvm).lower(),
            }
            if max_latency_ms > 0:
                data["max_latency_ms"] = max_latency_ms
            if max_memory_mb > 0:
                data["max_memory_mb"] = max_memory_mb
            
            resp = requests.post(f"{api_url}/deploy", data=data, files=files, timeout=120)
            
            if resp.status_code == 200:
                result = resp.json()
                st.session_state["pipeline_result"] = result
                st.success("✅ Optimization & benchmark pipeline completed successfully!")
            else:
                st.error(f"Pipeline failed (Status {resp.status_code}): {resp.text}")
        except Exception as err:
            st.error(f"Failed to communicate with backend: {err}")

# Render Results from Session State
if "pipeline_result" in st.session_state:
    res = st.session_state["pipeline_result"]
    winner = res.get("winner")
    all_results = res.get("all_results", [])
    
    st.markdown("## 🏆 Deployment Recommendation & Benchmark Results")
    
    split = res.get("data_split") or {}
    if split.get("test_samples"):
        st.caption(
            f"Calibrated on {split['calibration_samples']} images, "
            f"accuracy tested on {split['test_samples']} others."
        )
    for warning in res.get("warnings", []):
        st.warning(warning)

    if winner:
        st.markdown(f"""
        <div class="winner-card">
            <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #a5b4fc; font-weight: 700;">
                ⭐ Recommended Winner Pipeline
            </span>
            <h2 style="margin: 0.3rem 0; color: #ffffff;">{winner.get('name')}</h2>
            <p style="color: #cbd5e1; font-size: 0.85rem; margin-bottom: 1rem;">
                Target: <strong>{res.get('hardware_target', '').upper()}</strong> | Priority: <strong>{res.get('constraints', {}).get('priority', '').upper()}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        wc1, wc2, wc3, wc4 = st.columns(4)
        lat = winner.get("estimated_latency_p50_ms")
        fps = winner.get("throughput_fps")
        mem = winner.get("estimated_memory_mb")
        sz = winner.get("model_size_mb")
        
        wc1.metric("Est. Latency on Target", f"~{lat:.1f} ms" if lat is not None else "N/A")
        wc2.metric("Throughput", f"{fps:.1f} FPS" if fps else "N/A")
        wc3.metric("Est. Memory on Target", f"~{mem:.2f} MB" if mem is not None else "N/A")
        wc4.metric("Model Size", f"{sz:.2f} MB" if sz else "N/A")
        
        download_url = res.get("download_url")
        if download_url:
            full_dl_url = f"{api_url}{download_url}"
            st.markdown(f"""
            <a href="{full_dl_url}" target="_blank" style="text-decoration: none;">
                <button style="width: 100%; background: linear-gradient(90deg, #10b981, #059669); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 0.75rem; font-weight: 700; font-size: 1rem; cursor: pointer; margin-top: 0.5rem; margin-bottom: 1.5rem;">
                    📦 Download Production Deployment Package (.zip)
                </button>
            </a>
            """, unsafe_allow_html=True)
    
    # Candidate Comparison Section
    st.subheader("📊 Candidate Pipelines Comparison")
    if all_results:
        table_rows = []
        for r in all_results:
            top1 = None
            if r.get("accuracy") and r["accuracy"].get("top1_accuracy") is not None:
                top1 = f"{r['accuracy']['top1_accuracy'] * 100:.1f}%"
            
            table_rows.append({
                "Strategy": r.get("name"),
                "Framework": r.get("framework", "onnxruntime"),
                "Status": "✅ Success" if r.get("success") else f"❌ Failed ({r.get('error')})",
                "Host Latency p50 (ms)": r.get("latency_p50_ms"),
                "Est. Latency on Target (ms)": r.get("estimated_latency_p50_ms"),
                "Throughput (FPS)": r.get("throughput_fps"),
                "Est. Memory (MB)": r.get("estimated_memory_mb"),
                "Model Size (MB)": r.get("model_size_mb"),
                "Top-1 Accuracy": top1 or "--",
            })
        
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Comparison Bar Charts
        valid_df = df[df["Status"].str.contains("Success")].copy()
        if not valid_df.empty:
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.caption("Estimated Median Latency on Target (ms) — Lower is better")
                st.bar_chart(valid_df.set_index("Strategy")["Est. Latency on Target (ms)"])
            with chart_col2:
                st.caption("Throughput (FPS) — Higher is better")
                st.bar_chart(valid_df.set_index("Strategy")["Throughput (FPS)"])
