"""
Watchdog API - FastAPI Backend
Orchestrates all three agents: Technician, Auditor, CFO

Endpoints:
- POST /audit/run - Run full audit (streaming)
- GET /audit/health - Health check
- POST /audit/quick - Quick audit (limited records)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import asyncio
from datetime import datetime

from technician_agent import TechnicianAgent
from auditor_agent import AuditorAgent
from cfo_agent import CFOAgent

app = FastAPI(
    title="Watchdog - Account Health & Readiness Agent",
    description="AI-powered ad tech audit system that detects invisible setup issues",
    version="1.0.0"
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuditRequest(BaseModel):
    """Request model for audit endpoint."""
    limit: Optional[int] = None  # Limit records to process (for demo)
    include_cfo_narrative: bool = True


class AuditResponse(BaseModel):
    """Response model for audit results."""
    health_score: int
    total_findings: int
    p0_critical: int
    p1_high: int
    p2_medium: int
    financial_risk: dict
    executive_narrative: str
    findings: List[dict]
    reasoning_steps: List[dict]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Watchdog - Account Health Agent",
        "status": "running",
        "version": "1.0.0",
        "agents": ["Technician", "Auditor", "CFO"]
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "technician": "ready",
            "auditor": "ready",
            "cfo": "ready"
        }
    }


async def stream_audit_events(limit: Optional[int] = None, include_cfo: bool = True):
    """Generator that streams audit events as SSE."""
    all_findings = []
    all_reasoning = []
    
    # Initialize agents
    technician = TechnicianAgent()
    auditor = AuditorAgent()
    cfo = CFOAgent()
    
    # Stream Technician Agent events
    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Technician', 'message': 'Starting Technician Agent...'})}\n\n"
    await asyncio.sleep(0.1)
    
    for event in technician.run_audit(limit=limit):
        if event.get("type") == "finding":
            all_findings.append(event["data"])
            yield f"data: {json.dumps({'type': 'finding', 'agent': 'Technician', 'data': event['data']})}\n\n"
        else:
            all_reasoning.append(event)
            yield f"data: {json.dumps({'type': 'step', 'agent': 'Technician', 'step': event.get('step', '')})}\n\n"
        await asyncio.sleep(0.05)  # Small delay for streaming effect
    
    technician_summary = technician.get_summary()
    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Technician', 'summary': {'findings': technician_summary['total_findings']}})}\n\n"
    await asyncio.sleep(0.2)
    
    # Stream Auditor Agent events
    yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'Auditor', 'message': 'Starting Auditor Agent...'})}\n\n"
    await asyncio.sleep(0.1)
    
    for event in auditor.run_audit(limit=limit):
        if event.get("type") == "finding":
            all_findings.append(event["data"])
            yield f"data: {json.dumps({'type': 'finding', 'agent': 'Auditor', 'data': event['data']})}\n\n"
        else:
            all_reasoning.append(event)
            yield f"data: {json.dumps({'type': 'step', 'agent': 'Auditor', 'step': event.get('step', '')})}\n\n"
        await asyncio.sleep(0.05)
    
    auditor_summary = auditor.get_summary()
    yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'Auditor', 'summary': {'findings': auditor_summary['total_findings']}})}\n\n"
    await asyncio.sleep(0.2)
    
    # Stream CFO Agent events
    if include_cfo:
        yield f"data: {json.dumps({'type': 'agent_start', 'agent': 'CFO', 'message': 'Starting CFO Agent...'})}\n\n"
        await asyncio.sleep(0.1)
        
        cfo_report = None
        for event in cfo.analyze(all_findings):
            if event.get("type") == "cfo_report":
                cfo_report = event["data"]
                yield f"data: {json.dumps({'type': 'cfo_report', 'data': cfo_report})}\n\n"
            else:
                all_reasoning.append(event)
                yield f"data: {json.dumps({'type': 'step', 'agent': 'CFO', 'step': event.get('step', '')})}\n\n"
            await asyncio.sleep(0.05)
        
        yield f"data: {json.dumps({'type': 'agent_complete', 'agent': 'CFO', 'summary': {'health_score': cfo_report['health_score'] if cfo_report else 0}})}\n\n"
    
    # Final summary
    p0_count = len([f for f in all_findings if f.get('priority') == 'P0'])
    p1_count = len([f for f in all_findings if f.get('priority') == 'P1'])
    p2_count = len([f for f in all_findings if f.get('priority') == 'P2'])
    
    final_summary = {
        "type": "audit_complete",
        "total_findings": len(all_findings),
        "p0_critical": p0_count,
        "p1_high": p1_count,
        "p2_medium": p2_count,
        "health_score": cfo_report['health_score'] if cfo_report else 100 - (p0_count * 10 + p1_count * 5 + p2_count * 2)
    }
    yield f"data: {json.dumps(final_summary)}\n\n"


@app.post("/audit/stream")
async def run_audit_stream(request: AuditRequest):
    """
    Run full audit with streaming response (SSE).
    Use this for real-time UI updates.
    """
    return StreamingResponse(
        stream_audit_events(request.limit, request.include_cfo_narrative),
        media_type="text/event-stream"
    )


@app.post("/audit/run", response_model=AuditResponse)
async def run_audit(request: AuditRequest):
    """
    Run full audit and return complete results.
    Use this for non-streaming clients.
    """
    all_findings = []
    all_reasoning = []
    
    # Run Technician Agent
    technician = TechnicianAgent()
    for event in technician.run_audit(limit=request.limit):
        if event.get("type") == "finding":
            all_findings.append(event["data"])
        else:
            all_reasoning.append(event)
    
    # Run Auditor Agent
    auditor = AuditorAgent()
    for event in auditor.run_audit(limit=request.limit):
        if event.get("type") == "finding":
            all_findings.append(event["data"])
        else:
            all_reasoning.append(event)
    
    # Run CFO Agent
    cfo_report = None
    if request.include_cfo_narrative:
        cfo = CFOAgent()
        for event in cfo.analyze(all_findings):
            if event.get("type") == "cfo_report":
                cfo_report = event["data"]
            else:
                all_reasoning.append(event)
    
    # Build response
    p0_count = len([f for f in all_findings if f.get('priority') == 'P0'])
    p1_count = len([f for f in all_findings if f.get('priority') == 'P1'])
    p2_count = len([f for f in all_findings if f.get('priority') == 'P2'])
    
    return AuditResponse(
        health_score=cfo_report['health_score'] if cfo_report else 100 - (p0_count * 10 + p1_count * 5 + p2_count * 2),
        total_findings=len(all_findings),
        p0_critical=p0_count,
        p1_high=p1_count,
        p2_medium=p2_count,
        financial_risk=cfo_report['financial_risk'] if cfo_report else {"daily_spend_at_risk": 0, "monthly_risk": 0},
        executive_narrative=cfo_report['executive_narrative'] if cfo_report else "No narrative generated.",
        findings=all_findings,
        reasoning_steps=all_reasoning
    )


@app.post("/audit/quick")
async def quick_audit():
    """
    Quick audit with limited records (for demo).
    Processes only 50 records for speed.
    """
    request = AuditRequest(limit=50, include_cfo_narrative=True)
    return await run_audit(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
