"""
CFO Agent - LLM that writes the "Scary Story" narrative
Uses Google Gemini (free tier) to generate executive-level risk narratives
"""
import os
from datetime import datetime
from typing import List, Generator
from pathlib import Path

# Load .env file (optional)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
except:
    pass

# Preferred models in order
PREFERRED_MODELS = [
    "models/gemini-flash-latest", # Generic alias
    "models/gemini-pro-latest",
    "models/gemini-exp-1206", # Experimental Flash 2.0
    "models/gemini-2.5-flash", 
]


def get_secret(key: str, default: str = None) -> str:
    """Get secret from Streamlit secrets or environment."""
    # Try Streamlit secrets first
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    # Fall back to environment
    return os.getenv(key, default)


def get_genai_client(api_key: str):
    """Lazy load the AI client (Groq or Gemini)."""
    # Try Groq first if available
    if api_key.startswith("gsk_"):
        try:
            from groq import Groq
            return Groq(api_key=api_key), "groq"
        except ImportError:
            print("❌ Groq library not installed. Run `pip install groq`.")
        except Exception as e:
            print(f"❌ Error initializing Groq: {e}")
            
    # Try old package (google-generativeai) first as it is more stable for API keys
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai, False
    except ImportError:
        pass
    except Exception as e:
        print(f"Error initializing google.generativeai: {e}")
    
    # Try new package (google-genai)
    try:
        from google import genai
        return genai.Client(api_key=api_key), True
    except ImportError:
        pass
    except Exception as e:
        print(f"Error initializing google-genai: {e}")
    
    return None, False


class CFOAgent:
    """Agent that uses LLM to generate executive-level risk narratives."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or get_secret("GROQ_API_KEY") or get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        self.client = None
        self.client_type = None  # 'groq', 'gemini_new', 'gemini_old'
        self.model = PREFERRED_MODELS[0] if PREFERRED_MODELS else None
        self.reasoning_steps = []
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of the client."""
        if self._initialized:
            return
        self._initialized = True
        
        if self.api_key:
            self.client, self.client_type = get_genai_client(self.api_key)
            if not self.client:
                print("❌ CFO Agent: Failed to initialize AI client.")
                print(f"✅ CFO Agent: Initialized client (Type: {self.client_type})")
                if self.client_type == "groq":
                    self.model = "llama-3.1-8b-instant"
    
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
                    total_daily_spend_at_risk += float(daily_spend) * 0.5
        
        monthly_risk = total_daily_spend_at_risk * 30
        
        return {
            "daily_spend_at_risk": round(total_daily_spend_at_risk, 2),
            "monthly_risk": round(monthly_risk, 2),
            "critical_issues": critical_issues,
            "high_issues": high_issues
        }
    
    def _calculate_health_score(self, findings: List[dict], total_records: int = 1000) -> int:
        """Calculate account health score (0-100)."""
        if total_records == 0:
            return 100
        
        p0_count = len([f for f in findings if f.get('priority') == 'P0'])
        p1_count = len([f for f in findings if f.get('priority') == 'P1'])
        p2_count = len([f for f in findings if f.get('priority') == 'P2'])
        
        weighted_errors = (p0_count * 3) + (p1_count * 2) + (p2_count * 1)
        max_weighted_errors = total_records * 3
        error_rate = weighted_errors / max_weighted_errors if max_weighted_errors > 0 else 0
        
        import math
        score = int(100 * math.exp(-3 * error_rate))
        return max(0, min(100, score))
    
    def _try_generate(self, prompt: str) -> str:
        """Try to generate content with fallback through models."""
        self._ensure_initialized()
        
        if not self.client:
            return None
            
        if self.client_type == "groq":
             # Use Groq
             try:
                 chat_completion = self.client.chat.completions.create(
                     messages=[
                         {"role": "system", "content": "You are a CFO Risk Auditor. Be concise and dramatic."},
                         {"role": "user", "content": prompt}
                     ],
                     model="llama-3.1-8b-instant",
                     temperature=0.7,
                 )
                 self.model = "groq-llama-3.1"
                 return chat_completion.choices[0].message.content
             except Exception as e:
                 print(f"❌ Groq Error: {e}")
                 return None

        # GEMINI Logic
        for model_name in PREFERRED_MODELS:
            try:
                if self.client_type == True: # is_new_api
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    if response and response.text:
                        self.model = model_name
                        return response.text.strip()
                else:
                    # Old API
                    model = self.client.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    if response and response.text:
                        self.model = model_name
                        return response.text.strip()
            except Exception:
                 # Silently continue to next model on error (e.g. 429, 404)
                 continue
        
        return None
    
    def _generate_narrative(self, findings: List[dict], financial_risk: dict, batch_id: int = None, batch_size: int = None) -> tuple[str, bool]:
        """Generate narrative using LLM or fallback to template. Returns (text, used_fallback)."""
        # Prepare prompt
        top_findings = findings[:5]
        findings_text = "\n".join([
            f"- {f.get('priority', 'P?')}: {f.get('issue', 'Unknown')} "
            f"(Daily spend: ${f.get('daily_spend', 0):.2f})"
            for f in top_findings
        ])
        
        batch_context = f"IN BATCH #{batch_id} ({batch_size} RECORDS)" if batch_id else "IN THIS AUDIT"
        
        prompt = f"""You are a ruthless CFO reviewing an ad tech account audit. 
Explain why these issues are burning cash and demand immediate action.

CONTEXT: {batch_context}

FINANCIAL RISK:
- Daily spend at risk: ${financial_risk['daily_spend_at_risk']:,.2f}
- Monthly exposure: ${financial_risk['monthly_risk']:,.2f}
- Critical issues (P0): {financial_risk['critical_issues']}
- High priority issues (P1): {financial_risk['high_issues']}

TOP ISSUES:
{findings_text}

Write a 3-4 sentence SCARY executive summary. Be direct and create urgency. Mention the Batch Size explicitly if provided.
Do NOT use bullet points. Write in prose. Be dramatic but factual."""

        # Try LLM
        result = self._try_generate(prompt)
        if result:
            return result, False
            
        # Fallback to template
        return self._generate_fallback(findings, financial_risk, batch_id, batch_size), True
    
    def _generate_fallback(self, findings: List[dict], financial_risk: dict, batch_id: int = None, batch_size: int = None) -> str:
        """Template-based narrative when LLM is unavailable."""
        import random
        
        context = f"in Batch #{batch_id} ({batch_size} records)" if batch_id else "in this audit"
        
        p0_count = len([f for f in findings if f.get('priority') == 'P0'])
        p1_count = len([f for f in findings if f.get('priority') == 'P1'])
        
        # Dynamic vocab
        openers = [
            "This account is hemorrhaging money due to invisible tracking failures.",
            "Immediate intervention required: Critical signal loss is draining budget.",
            "We are seeing a catastrophic disconnect between spend and measurement.",
            "Audit reveals severe data governance gaps that are killing ROI.",
            "Blind spots in the tracking setup are causing significant waste."
        ]
        
        middles = [
            f"Our audit {context} identified {p0_count} critical and {p1_count} high-priority issues that are actively degrading campaign performance.",
            f"We found {p0_count} P0 blockers and {p1_count} P1 warnings. These are not false alarms.",
            f"With {p0_count + p1_count} confirmed tracking failures, the bidding strategy is effectively flying blind.",
            f"The system detected {p0_count} critical breakages. This is impacting ${financial_risk['daily_spend_at_risk']:,.2f} of daily spend."
        ]
        
        closers = [
            "The bidding algorithms are optimizing towards broken signals, effectively spending budget on ghost conversions.",
            "Every hour of inaction compounds the waste. This requires urgent remediation.",
            "You are paying premium CPMs for broken data. Fix this immediately.",
            "This is a P0 emergency. Pause optimization until tracking is restored."
        ]
        
        severity = "CRITICAL" if p0_count > 0 else "HIGH RISK"
        
        return f"{severity} ALERT: {random.choice(openers)}\n\n{random.choice(middles)} With ${financial_risk['daily_spend_at_risk']:,.2f} in daily spend at risk, the monthly exposure reaches ${financial_risk['monthly_risk']:,.2f}.\n\n{random.choice(closers)}"
    
    def analyze(self, findings: List[dict], batch_id: int = None, batch_size: int = None, total_records: int = 1000) -> Generator[dict, None, None]:
        """Analyze findings and generate CFO report."""
        self.reasoning_steps = []
        
        batch_context = f" (Batch #{batch_id}, {batch_size} records)" if batch_id else ""
        yield self._log_step(f"💰 CFO Agent analyzing financial impact{batch_context}...")
        
        yield self._log_step("📊 Calculating total spend at risk...")
        financial_risk = self._calculate_financial_risk(findings)
        yield self._log_step(f"   Daily spend at risk: ${financial_risk['daily_spend_at_risk']:,.2f}")
        yield self._log_step(f"   Monthly exposure: ${financial_risk['monthly_risk']:,.2f}")
        
        yield self._log_step("🏥 Computing account health score...")
        health_score = self._calculate_health_score(findings, total_records)
        yield self._log_step(f"   Health Score: {health_score}/100")
        
        yield self._log_step("✍️ Generating executive narrative...")
        
        # Check if LLM is available
        self._ensure_initialized()
        if self.client and self.model:
            title = "Groq Llama 3" if self.client_type == "groq" else f"Gemini AI ({self.model})"
            yield self._log_step(f"   🤖 Using {title}...")
        else:
            yield self._log_step("   📝 Using template narrative...")
        
        narrative, used_fallback = self._generate_narrative(findings, financial_risk, batch_id, batch_size)
        
        if used_fallback and self.client:
             yield self._log_step(f"   ⚠️ Connection to LLM failed. Switched to Pseudo Rules Engine.")
        elif not used_fallback:
             yield self._log_step(f"   ✅ Generated narrative via {self.model}.")
        
        yield self._log_step("✅ CFO Agent analysis complete.")
        
        yield {
            "type": "cfo_report",
            "data": {
                "health_score": health_score,
                "financial_risk": financial_risk,
                "executive_narrative": narrative,
                "batch_id": batch_id,
                "batch_size": batch_size,
                "total_findings": len(findings),
                "p0_count": len([f for f in findings if f.get('priority') == 'P0']),
                "p1_count": len([f for f in findings if f.get('priority') == 'P1']),
                "p2_count": len([f for f in findings if f.get('priority') == 'P2']),
                "reasoning_steps": self.reasoning_steps
            }
        }


if __name__ == "__main__":
    findings = [
        {'priority': 'P0', 'issue': 'Dead Pixel', 'daily_spend': 5000},
        {'priority': 'P1', 'issue': 'ID Mismatch', 'daily_spend': 2000}
    ]
    
    agent = CFOAgent()
    for event in agent.analyze(findings):
        if event.get('type') == 'cfo_report':
            print(f"Health Score: {event['data']['health_score']}")
            print(f"Narrative: {event['data']['executive_narrative'][:200]}...")
        else:
            print(event.get('step', ''))
