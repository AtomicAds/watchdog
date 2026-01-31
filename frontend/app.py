"""
Watchdog UI - Streamlit Frontend
Glass Box Reasoning Display with Real-time Agent Updates

Features:
- Real-time streaming of agent reasoning steps
- Priority-coded findings display
- Health score visualization
- Executive narrative display
- Fix recommendations
"""
import streamlit as st
import time
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path

# Add backend to path for direct imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Lazy imports to avoid startup issues
TechnicianAgent = None
AuditorAgent = None
CFOAgent = None

def load_agents():
    """Lazy load agents to avoid import issues during startup."""
    global TechnicianAgent, AuditorAgent, CFOAgent
    if TechnicianAgent is None:
        from technician_agent import TechnicianAgent as TA
        from auditor_agent import AuditorAgent as AA
        from cfo_agent import CFOAgent as CA
        TechnicianAgent = TA
        AuditorAgent = AA
        CFOAgent = CA

# Page config
st.set_page_config(
    page_title="Watchdog - Account Health Agent",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .finding-p0 {
        background-color: #ff4b4b;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: white;
    }
    .finding-p1 {
        background-color: #ffa500;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: black;
    }
    .finding-p2 {
        background-color: #00cc00;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: black;
    }
    .agent-log {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        background-color: #1a1a2e;
        padding: 10px;
        border-radius: 5px;
        color: #00ff00;
        max-height: 400px;
        overflow-y: auto;
    }
    .health-score-good {
        color: #00ff00;
        font-size: 48px;
        font-weight: bold;
    }
    .health-score-warning {
        color: #ffa500;
        font-size: 48px;
        font-weight: bold;
    }
    .health-score-critical {
        color: #ff4b4b;
        font-size: 48px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e1e2e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def get_health_score_color(score: int) -> str:
    """Get color class based on health score."""
    if score >= 70:
        return "health-score-good"
    elif score >= 40:
        return "health-score-warning"
    else:
        return "health-score-critical"


def display_finding(finding: dict):
    """Display a single finding with appropriate styling."""
    priority = finding.get('priority', 'P2')
    priority_label = finding.get('priority_label', 'UNKNOWN')
    
    if priority == 'P0':
        icon = "🔴"
        bg_color = "#ff4b4b"
        text_color = "white"
    elif priority == 'P1':
        icon = "🟡"
        bg_color = "#ffa500"
        text_color = "black"
    else:
        icon = "🟢"
        bg_color = "#00cc00"
        text_color = "black"
    
    with st.expander(f"{icon} {priority} {priority_label}: {finding.get('issue', 'Unknown Issue')}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Reasoning Chain:**")
            reasoning = finding.get('reasoning', [])
            for i, step in enumerate(reasoning, 1):
                st.markdown(f"{i}. {step}")
            
            st.markdown("**Technical Proof:**")
            st.code(finding.get('technical_proof', 'N/A'))
        
        with col2:
            st.markdown("**Details:**")
            st.write(f"Agent: {finding.get('agent', 'Unknown')}")
            st.write(f"Check: {finding.get('check', 'Unknown')}")
            if finding.get('daily_spend'):
                st.write(f"Daily Spend: ${finding.get('daily_spend', 0):,.2f}")
            if finding.get('advertiser_id'):
                st.write(f"Advertiser: {finding.get('advertiser_id')}")
            
            st.markdown("**Recommendation:**")
            st.info(finding.get('recommendation', 'Review and fix'))


def run_audit_with_streaming(limit: int = 100):
    """Run the audit with continuous streaming display."""
    
    # Load agents (lazy loading)
    load_agents()
    
    # Initialize session state for persistent tracking across re-runs
    if 'findings' not in st.session_state:
        st.session_state.findings = []
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Agent Reasoning (Live Stream)")
        log_container = st.empty()
        
    with col2:
        st.subheader("📊 Live Stats")
        stats_container = st.empty()
    
    st.divider()
    
    # Container for live findings
    findings_container = st.container()
    
    # Initialize agents
    technician = TechnicianAgent()
    auditor = AuditorAgent()
    cfo = CFOAgent()
    
    def update_logs(message: str):
        """Append log and handle rolling window."""
        st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        # Keep last 100 logs
        if len(st.session_state.logs) > 100:
            st.session_state.logs.pop(0)
        log_container.markdown(f"<div class='agent-log'>{'<br>'.join(st.session_state.logs[-20:])}</div>", unsafe_allow_html=True)

    def update_stats(findings_list):
        p0 = len([f for f in findings_list if f.get('priority') == 'P0'])
        p1 = len([f for f in findings_list if f.get('priority') == 'P1'])
        p2 = len([f for f in findings_list if f.get('priority') == 'P2'])
        stats_container.markdown(f"""
        **Total Findings:** {len(findings_list)}  
        🔴 P0 Critical: {p0}  
        🟡 P1 High: {p1}  
        🟢 P2 Medium: {p2}
        """)

    update_logs("🚀 Entering Continuous Interleaved Mode...")
    progress_bar = st.progress(0, text="Initializing Agents...")
    
    # Generators
    tech_gen = technician.run_audit(limit=limit, batch_size=5)
    audit_gen = auditor.run_audit(limit=limit, batch_size=5)
    
    # Infinite Interleaved Loop
    cycle_count = 1
    
    while True:
        # --- Technician Agent Turn ---
        try:
            while True:
                event = next(tech_gen)
                event_type = event.get("type")
                
                if event_type == "finding":
                    st.session_state.findings.insert(0, event["data"]) # Add to top
                    # Display finding if P0 or P1
                    if event["data"].get("priority") in ["P0", "P1"]:
                        with findings_container:
                            display_finding(event["data"])
                
                elif event_type == "batch_start":
                    bid = event.get("batch_id")
                    tot = event.get("total_batches")
                    progress_bar.progress(bid / tot, text=f"🔧 Technician: Analyzing Batch {bid}/{tot}...")
                
                elif event.get("step"):
                    update_logs(event["step"])
                
                elif event_type == "batch_complete":
                     # Switch to Auditor after one batch
                     break
                     
                time.sleep(0.05)
                
        except StopIteration:
            update_logs(f"Note: Technician Agent completed cycle {cycle_count}. Restarting stream...")
            tech_gen = technician.run_audit(limit=limit, batch_size=5)
            time.sleep(0.5)
        
        # --- Auditor Agent Turn ---
        try:
             while True:
                event = next(audit_gen)
                event_type = event.get("type")
                
                if event_type == "finding":
                    st.session_state.findings.insert(0, event["data"])
                    if event["data"].get("priority") in ["P0", "P1"]:
                        with findings_container:
                            display_finding(event["data"])

                elif event_type == "batch_start":
                    bid = event.get("batch_id")
                    tot = event.get("total_batches")
                    progress_bar.progress(bid / tot, text=f"📋 Auditor: Analyzing Batch {bid}/{tot}...")
                
                elif event.get("step"):
                    update_logs(event["step"])
                    
                elif event_type == "batch_complete":
                     break
                     
                time.sleep(0.05)
        
        except StopIteration:
            update_logs(f"Note: Auditor Agent completed cycle {cycle_count}. Restarting stream...")
            audit_gen = auditor.run_audit(limit=limit, batch_size=5)
            cycle_count += 1
            time.sleep(0.5)

        # Update global stats
        # Cap session state to 200 findings to prevent memory explosion
        if len(st.session_state.findings) > 200:
             st.session_state.findings = st.session_state.findings[:200]
        
        update_stats(st.session_state.findings)
        time.sleep(0.1)


def main():
    # Header
    st.title("🐕 Watchdog")
    st.markdown("### Account Health & Readiness Agent")
    st.markdown("*Continuously evaluates ad accounts for invisible setup, tracking, and measurement issues*")
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        limit = st.slider(
            "Records to Analyze",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of records to process from each data source"
        )
        
        st.divider()
        
        st.markdown("### 🤖 Agents")
        st.markdown("""
        - **Technician**: Decision tree walker (DV360/GTM/Website)
        - **Auditor**: Governance rules checker (GA4)
        - **CFO**: Financial impact narrator (LLM)
        """)
        
        st.divider()
        
        st.markdown("### 📁 Data Sources")
        st.markdown("""
        - DV360 Audit Data (1000 rows)
        - GTM Tag Data (1000 rows)
        - GA4 Property Data (1000 rows)
        - Website Scan Data (1000 rows)
        """)
    
    # Main content
    run_button = st.button("🚀 Run Account Audit", type="primary", use_container_width=True)
    
    if run_button:
        with st.spinner("Running audit..."):
            findings, cfo_report = run_audit_with_streaming(limit=limit)
        
        st.divider()
        
        # Results Dashboard
        if cfo_report:
            st.header("📊 Audit Results")
            
            # Health Score and Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                health_score = cfo_report.get('health_score', 0)
                score_color = get_health_score_color(health_score)
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='{score_color}'>{health_score}</div>
                    <div>Health Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "🔴 Critical Issues",
                    cfo_report.get('p0_count', 0),
                    delta=None
                )
            
            with col3:
                st.metric(
                    "🟡 High Priority",
                    cfo_report.get('p1_count', 0),
                    delta=None
                )
            
            with col4:
                monthly_risk = cfo_report.get('financial_risk', {}).get('monthly_risk', 0)
                st.metric(
                    "💰 Monthly Risk",
                    f"${monthly_risk:,.0f}",
                    delta=None
                )
            
            st.divider()
            
            # Executive Narrative
            st.header("📝 Executive Summary")
            narrative = cfo_report.get('executive_narrative', 'No narrative generated.')
            st.error(narrative)  # Using error box for dramatic effect
            
            st.divider()
        
        # Findings List
        if findings:
            st.header(f"🔍 Findings ({len(findings)} issues)")
            
            # Filter tabs
            tab1, tab2, tab3, tab4 = st.tabs(["🔴 Critical (P0)", "🟡 High (P1)", "🟢 Medium (P2)", "📋 All"])
            
            with tab1:
                p0_findings = [f for f in findings if f.get('priority') == 'P0']
                if p0_findings:
                    for finding in p0_findings:
                        display_finding(finding)
                else:
                    st.success("No critical issues found!")
            
            with tab2:
                p1_findings = [f for f in findings if f.get('priority') == 'P1']
                if p1_findings:
                    for finding in p1_findings:
                        display_finding(finding)
                else:
                    st.success("No high priority issues found!")
            
            with tab3:
                p2_findings = [f for f in findings if f.get('priority') == 'P2']
                if p2_findings:
                    for finding in p2_findings:
                        display_finding(finding)
                else:
                    st.success("No medium priority issues found!")
            
            with tab4:
                for finding in findings:
                    display_finding(finding)
            
            st.divider()
            
            # Export options
            st.header("📤 Export")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copy Findings to Clipboard"):
                    findings_text = json.dumps(findings, indent=2)
                    st.code(findings_text[:500] + "..." if len(findings_text) > 500 else findings_text)
                    st.success("Findings displayed above - copy manually")
            
            with col2:
                findings_df = pd.DataFrame(findings)
                csv = findings_df.to_csv(index=False)
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    "watchdog_findings.csv",
                    "text/csv"
                )
    
    else:
        # Show placeholder content when not running
        st.info("👆 Click 'Run Account Audit' to start analyzing your ad accounts for setup issues.")
        
        st.markdown("""
        ### What Watchdog Checks:
        
        **🔧 Technical Issues (Technician Agent)**
        - Missing or dead Floodlight pixels
        - Cookie consent blocking data
        - GTM and DV360 ID mismatches
        - Counting method discrepancies
        - Blocked network calls
        
        **📋 Governance Issues (Auditor Agent)**
        - PII in URL parameters (GDPR risk)
        - Data retention settings
        - Google Signals configuration
        - Campaign naming conventions
        - Referral exclusion lists
        - Consent Mode status
        
        **💰 Financial Impact (CFO Agent)**
        - Calculates daily spend at risk
        - Generates executive narratives
        - Prioritizes fixes by business impact
        """)


if __name__ == "__main__":
    main()
