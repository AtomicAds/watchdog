"""
CFO Agent - LLM that writes the "Scary Story" narrative
Uses Google Gemini (free tier) to generate executive-level risk narratives

This agent takes findings from Technician and Auditor agents and:
1. Calculates total financial risk
2. Generates executive-friendly narratives
3. Prioritizes by business impact (not just technical severity)
4. Creates a "Health Score" for the account
"""
import os
from datetime import datetime
from typing import List, Generator
import json
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    # Load from project root
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass

# Also try to load from Streamlit secrets (for cloud deployment)
def get_secret(key: str, default: str = None) -> str:
    """Get secret from Streamlit secrets or environment."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return os.getenv(key, default)

# Try new google-genai package first, fall back to deprecated one
try:
    from google import genai
    GEMINI_AVAILABLE = True
    GEMINI_NEW_API = True
except ImportError:
    try:
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
        GEMINI_NEW_API = False
    except ImportError:
        GEMINI_AVAILABLE = False
        GEMINI_NEW_API = False

# Preferred models in order (best first, fallbacks after)
PREFERRED_MODELS = [
    "gemini-2.5-flash",      # Latest, best performance
    "gemini-2.5-pro",        # Pro version
    "gemini-2.0-flash",      # Previous gen
    "gemini-2.0-flash-lite", # Lite version
    "gemini-flash-latest",   # Generic latest
]


class CFOAgent:
    """Agent that uses LLM to generate executive-level risk narratives."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        self.client = None
        self.model = None
        self.reasoning_steps = []
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                if GEMINI_NEW_API:
                    # New google-genai package
                    self.client = genai.Client(api_key=self.api_key)
                    # Find the best available model
                    self.model = self._find_best_model()
                else:
                    # Old deprecated package
                    genai.configure(api_key=self.api_key)
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")
                self.client = None
                self.model = None
    
    def _find_best_model(self) -> str:
        """Find the best available model by testing each one."""
        if not self.client:
            return None
        
        for model_name in PREFERRED_MODELS:
            try:
                # Quick test to see if model works
                response = self.client.models.generate_content(
                    model=model_name,
                    contents="Hi"
                )
                if response and response.text:
                    print(f"Using model: {model_name}")
                    return model_name
            except Exception as e:
                # Model not available or rate limited, try next
                continue
        
        # No model worked, return default (will use fallback narrative)
        print("Warning: No Gemini model available, using template narrative")
        return None
    
    def _log_step(self, step: str) -> dict:
        """Log a reasoning step."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": "CFO",
            "step": step
        }
        self.reasoning_steps.append(log_entry)
        return log_entry
    
    def _calculate_financial_risk(self, findings: List[dict]) -> dict:
        """Calculate total financial risk from all findings."""
        total_daily_spend_at_risk = 0
        critical_issues = 0
        high_issues = 0
        
        for finding in findings:
            if finding.get('priority') == 'P0':
                critical_issues += 1
                daily_spend = finding.get('daily_spend', 0)
                if daily_spend:
                    total_daily_spend_at_risk += float(daily_spend)
            elif finding.get('priority') == 'P1':
                high_issues += 1
                daily_spend = finding.get('daily_spend', 0)
                if daily_spend:
                    total_daily_spend_at_risk += float(daily_spend) * 0.5  # 50% risk for high issues
        
        monthly_risk = total_daily_spend_at_risk * 30
        
        return {
            "daily_spend_at_risk": round(total_daily_spend_at_risk, 2),
            "monthly_risk": round(monthly_risk, 2),
            "critical_issues": critical_issues,
            "high_issues": high_issues
        }
    
    def _calculate_health_score(self, findings: List[dict], total_records: int = 1000) -> int:
        """
        Calculate account health score (0-100).
        100 = Perfect health, 0 = Complete disaster
        
        Uses a percentage-based weighted formula:
        - P0 issues count as 3x weight
        - P1 issues count as 2x weight
        - P2 issues count as 1x weight
        
        Score = 100 - (weighted_error_rate * 100)
        """
        if total_records == 0:
            return 100
        
        # Count issues by priority
        p0_count = len([f for f in findings if f.get('priority') == 'P0'])
        p1_count = len([f for f in findings if f.get('priority') == 'P1'])
        p2_count = len([f for f in findings if f.get('priority') == 'P2'])
        
        # Calculate weighted error count (P0 = 3x, P1 = 2x, P2 = 1x)
        weighted_errors = (p0_count * 3) + (p1_count * 2) + (p2_count * 1)
        
        # Max possible weighted errors (if all records were P0)
        max_weighted_errors = total_records * 3
        
        # Calculate error rate (0 to 1)
        error_rate = weighted_errors / max_weighted_errors if max_weighted_errors > 0 else 0
        
        # Apply a curve to make the score more meaningful
        # Using inverse exponential: more sensitive to small error rates
        import math
        # Score decreases faster initially, then slows down
        # This ensures even small issues affect the score visibly
        score = int(100 * math.exp(-3 * error_rate))
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        return score
    
    def _generate_narrative_with_llm(self, findings: List[dict], financial_risk: dict) -> str:
        """Generate a scary CFO narrative using Gemini."""
        if not self.model or (GEMINI_NEW_API and not self.client):
            return self._generate_fallback_narrative(findings, financial_risk)
        
        # Prepare the prompt
        top_findings = findings[:5]  # Top 5 issues
        findings_text = "\n".join([
            f"- {f.get('priority', 'P?')}: {f.get('issue', 'Unknown issue')} "
            f"(Daily spend: ${f.get('daily_spend', 0):.2f})"
            for f in top_findings
        ])
        
        prompt = f"""You are a ruthless CFO reviewing an ad tech account audit. 
You need to explain why these issues are burning cash and demand immediate action.

FINANCIAL RISK SUMMARY:
- Daily spend at risk: ${financial_risk['daily_spend_at_risk']:,.2f}
- Monthly exposure: ${financial_risk['monthly_risk']:,.2f}
- Critical issues (P0): {financial_risk['critical_issues']}
- High priority issues (P1): {financial_risk['high_issues']}

TOP ISSUES FOUND:
{findings_text}

Write a 3-4 sentence SCARY executive summary. Be direct, use financial language, and create urgency.
Focus on: What's broken, how much money is being wasted, and what happens if we don't fix it TODAY.
Do NOT use bullet points. Write in prose. Be dramatic but factual."""

        try:
            if GEMINI_NEW_API and self.client:
                # New google-genai API
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                return response.text.strip()
            elif self.model:
                # Old deprecated API
                response = self.model.generate_content(prompt)
                return response.text.strip()
            else:
                return self._generate_fallback_narrative(findings, financial_risk)
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._generate_fallback_narrative(findings, financial_risk)
    
    def _generate_fallback_narrative(self, findings: List[dict], financial_risk: dict) -> str:
        """Generate narrative without LLM (fallback)."""
        p0_count = len([f for f in findings if f.get('priority') == 'P0'])
        p1_count = len([f for f in findings if f.get('priority') == 'P1'])
        
        if p0_count > 0:
            severity = "CRITICAL"
            urgency = "immediate executive attention"
        elif p1_count > 0:
            severity = "HIGH RISK"
            urgency = "urgent remediation"
        else:
            severity = "MODERATE"
            urgency = "scheduled review"
        
        narrative = f"""
{severity} ACCOUNT HEALTH ALERT: This account is hemorrhaging money due to invisible tracking failures. 

Our audit identified {p0_count} critical and {p1_count} high-priority issues that are actively degrading campaign performance. 
With ${financial_risk['daily_spend_at_risk']:,.2f} in daily spend at risk, the monthly exposure reaches ${financial_risk['monthly_risk']:,.2f}.

The bidding algorithms are optimizing towards broken signals, effectively spending budget on ghost conversions. 
Every hour of inaction compounds the waste. This requires {urgency}.
""".strip()
        
        return narrative
    
    def analyze(self, findings: List[dict], total_records: int = 1000) -> Generator[dict, None, None]:
        """
        Analyze all findings and generate the CFO report.
        Yields events for streaming.
        """
        self.reasoning_steps = []
        
        yield self._log_step("💰 CFO Agent analyzing financial impact...")
        
        # Calculate financial risk
        yield self._log_step("📊 Calculating total spend at risk...")
        financial_risk = self._calculate_financial_risk(findings)
        yield self._log_step(f"   Daily spend at risk: ${financial_risk['daily_spend_at_risk']:,.2f}")
        yield self._log_step(f"   Monthly exposure: ${financial_risk['monthly_risk']:,.2f}")
        
        # Calculate health score
        yield self._log_step("🏥 Computing account health score...")
        health_score = self._calculate_health_score(findings, total_records)
        yield self._log_step(f"   Health Score: {health_score}/100")
        
        # Generate narrative
        yield self._log_step("✍️ Generating executive narrative...")
        
        if self.model and self.client:
            yield self._log_step(f"   🤖 Using Gemini AI ({self.model})...")
        elif self.model:
            yield self._log_step(f"   🤖 Using Gemini AI ({self.model})...")
        else:
            yield self._log_step("   📝 Using template-based narrative (no API key configured)...")
        
        narrative = self._generate_narrative_with_llm(findings, financial_risk)
        
        yield self._log_step("✅ CFO Agent analysis complete.")
        
        # Yield the final report
        report = {
            "type": "cfo_report",
            "data": {
                "health_score": health_score,
                "financial_risk": financial_risk,
                "executive_narrative": narrative,
                "total_findings": len(findings),
                "p0_count": len([f for f in findings if f.get('priority') == 'P0']),
                "p1_count": len([f for f in findings if f.get('priority') == 'P1']),
                "p2_count": len([f for f in findings if f.get('priority') == 'P2']),
                "reasoning_steps": self.reasoning_steps
            }
        }
        yield report
    
    def generate_fix_recommendations(self, findings: List[dict]) -> List[dict]:
        """Generate prioritized fix recommendations."""
        recommendations = []
        
        # Sort by priority (P0 first, then P1, then P2)
        priority_order = {'P0': 0, 'P1': 1, 'P2': 2}
        sorted_findings = sorted(findings, key=lambda x: priority_order.get(x.get('priority', 'P2'), 3))
        
        for i, finding in enumerate(sorted_findings[:10], 1):  # Top 10 recommendations
            rec = {
                "rank": i,
                "priority": finding.get('priority', 'P2'),
                "issue": finding.get('issue', 'Unknown'),
                "recommendation": finding.get('recommendation', 'Review and fix'),
                "impact": f"${finding.get('daily_spend', 0):.2f}/day at risk" if finding.get('daily_spend') else "Unknown impact"
            }
            recommendations.append(rec)
        
        return recommendations


if __name__ == "__main__":
    # Test with sample findings
    sample_findings = [
        {
            "priority": "P0",
            "issue": "Dead Pixel - No Recent Conversions",
            "daily_spend": 2500,
            "recommendation": "Check pixel placement on conversion page"
        },
        {
            "priority": "P0",
            "issue": "PII Detected - Email in URL",
            "daily_spend": 0,
            "recommendation": "Remove or hash email parameter"
        },
        {
            "priority": "P1",
            "issue": "Advertiser ID Mismatch",
            "daily_spend": 1500,
            "recommendation": "Align GTM and DV360 advertiser IDs"
        }
    ]
    
    agent = CFOAgent()
    for event in agent.analyze(sample_findings):
        if event.get("type") == "cfo_report":
            report = event["data"]
            print(f"\n🏥 Health Score: {report['health_score']}/100")
            print(f"💰 Monthly Risk: ${report['financial_risk']['monthly_risk']:,.2f}")
            print(f"\n📝 Executive Summary:\n{report['executive_narrative']}")
        else:
            print(event.get("step", ""))
