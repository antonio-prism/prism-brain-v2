"""
PRISM Brain Phase 2 - Client Data API Endpoints
FastAPI route registration for client CRUD operations.
Called from main.py via register_client_routes(app, get_session_context).
"""

from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from database.models import Client, ClientProcess, ClientRisk, ClientRiskAssessment


# ============== Pydantic Models ==============

class ClientCreate(BaseModel):
    name: str
    location: str = ""
    industry: str = ""
    revenue: float = 0
    employees: int = 0
    currency: str = "EUR"
    export_percentage: float = 0
    primary_markets: str = ""
    sectors: str = ""
    notes: str = ""

class ClientUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    revenue: Optional[float] = None
    employees: Optional[int] = None
    currency: Optional[str] = None
    export_percentage: Optional[float] = None
    primary_markets: Optional[str] = None
    sectors: Optional[str] = None
    notes: Optional[str] = None

class ProcessCreate(BaseModel):
    process_id: str
    process_name: str
    custom_name: str = ""
    category: str = ""
    criticality_per_day: float = 0
    notes: str = ""

class ProcessUpdate(BaseModel):
    custom_name: Optional[str] = None
    criticality_per_day: Optional[float] = None
    notes: Optional[str] = None

class RiskCreate(BaseModel):
    risk_id: str
    risk_name: str
    domain: str = ""
    category: str = ""
    probability: float = 0.5
    is_prioritized: bool = False
    notes: str = ""

class RiskUpdate(BaseModel):
    probability: Optional[float] = None
    is_prioritized: Optional[bool] = None
    notes: Optional[str] = None

class AssessmentCreate(BaseModel):
    process_id: int
    risk_id: int
    vulnerability: float = 0.5
    resilience: float = 0.3
    expected_downtime: int = 5
    notes: str = ""

class AssessmentUpdate(BaseModel):
    vulnerability: Optional[float] = None
    resilience: Optional[float] = None
    expected_downtime: Optional[int] = None
    notes: Optional[str] = None


# ============== Route Registration ==============

def register_client_routes(app, get_session_context):
    """Register all Phase 2 client CRUD endpoints on the FastAPI app."""

    # ---- Client CRUD ----

    @app.post("/api/v1/clients")
    async def create_client(client: ClientCreate):
        """Create a new client."""
        with get_session_context() as session:
            db_client = Client(
                name=client.name,
                location=client.location,
                industry=client.industry,
                revenue=client.revenue,
                employees=client.employees,
                currency=client.currency,
                export_percentage=client.export_percentage,
                primary_markets=client.primary_markets,
                sectors=client.sectors,
                notes=client.notes
            )
            session.add(db_client)
            session.flush()
            client_id = db_client.id
            return {"id": client_id, "message": "Client created successfully"}

    @app.get("/api/v1/clients")
    async def list_clients():
        """List all clients."""
        with get_session_context() as session:
            clients = session.query(Client).order_by(Client.updated_at.desc()).all()
            return {
                "total": len(clients),
                "clients": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "location": c.location,
                        "industry": c.industry,
                        "revenue": c.revenue,
                        "employees": c.employees,
                        "currency": c.currency,
                        "export_percentage": c.export_percentage,
                        "primary_markets": c.primary_markets,
                        "sectors": c.sectors,
                        "notes": c.notes,
                        "created_at": c.created_at.isoformat() if c.created_at else None,
                        "updated_at": c.updated_at.isoformat() if c.updated_at else None
                    }
                    for c in clients
                ]
            }

    @app.get("/api/v1/clients/{client_id}")
    async def get_client(client_id: int):
        """Get a specific client with summary counts."""
        with get_session_context() as session:
            client = session.query(Client).filter(Client.id == client_id).first()
            if not client:
                raise HTTPException(status_code=404, detail="Client not found")
            process_count = session.query(ClientProcess).filter(
                ClientProcess.client_id == client_id
            ).count()
            risk_count = session.query(ClientRisk).filter(
                ClientRisk.client_id == client_id
            ).count()
            assessment_count = session.query(ClientRiskAssessment).filter(
                ClientRiskAssessment.client_id == client_id
            ).count()
            return {
                "id": client.id,
                "name": client.name,
                "location": client.location,
                "industry": client.industry,
                "revenue": client.revenue,
                "employees": client.employees,
                "currency": client.currency,
                "export_percentage": client.export_percentage,
                "primary_markets": client.primary_markets,
                "sectors": client.sectors,
                "notes": client.notes,
                "created_at": client.created_at.isoformat() if client.created_at else None,
                "updated_at": client.updated_at.isoformat() if client.updated_at else None,
                "process_count": process_count,
                "risk_count": risk_count,
                "assessment_count": assessment_count
            }

    @app.put("/api/v1/clients/{client_id}")
    async def update_client(client_id: int, updates: ClientUpdate):
        """Update a client."""
        with get_session_context() as session:
            client = session.query(Client).filter(Client.id == client_id).first()
            if not client:
                raise HTTPException(status_code=404, detail="Client not found")
            update_data = updates.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(client, field, value)
            client.updated_at = datetime.utcnow()
            return {"message": "Client updated successfully"}

    @app.delete("/api/v1/clients/{client_id}")
    async def delete_client(client_id: int):
        """Delete a client and all associated data (cascades)."""
        with get_session_context() as session:
            client = session.query(Client).filter(Client.id == client_id).first()
            if not client:
                raise HTTPException(status_code=404, detail="Client not found")
            session.delete(client)
            return {"message": "Client deleted successfully"}

    # ---- Client Process Endpoints ----

    @app.post("/api/v1/clients/{client_id}/processes")
    async def add_client_process(client_id: int, process: ProcessCreate):
        """Add a business process to a client."""
        with get_session_context() as session:
            client = session.query(Client).filter(Client.id == client_id).first()
            if not client:
                raise HTTPException(status_code=404, detail="Client not found")
            db_process = ClientProcess(
                client_id=client_id,
                process_id=process.process_id,
                process_name=process.process_name,
                custom_name=process.custom_name,
                category=process.category,
                criticality_per_day=process.criticality_per_day,
                notes=process.notes
            )
            session.add(db_process)
            session.flush()
            return {"id": db_process.id, "message": "Process added successfully"}

    @app.get("/api/v1/clients/{client_id}/processes")
    async def list_client_processes(client_id: int):
        """Get all processes for a client."""
        with get_session_context() as session:
            processes = session.query(ClientProcess).filter(
                ClientProcess.client_id == client_id
            ).order_by(ClientProcess.criticality_per_day.desc()).all()
            return {
                "total": len(processes),
                "processes": [
                    {
                        "id": p.id,
                        "client_id": p.client_id,
                        "process_id": p.process_id,
                        "process_name": p.process_name,
                        "custom_name": p.custom_name,
                        "category": p.category,
                        "criticality_per_day": p.criticality_per_day,
                        "notes": p.notes,
                        "created_at": p.created_at.isoformat() if p.created_at else None
                    }
                    for p in processes
                ]
            }

    @app.put("/api/v1/clients/{client_id}/processes/{process_db_id}")
    async def update_client_process(client_id: int, process_db_id: int, updates: ProcessUpdate):
        """Update a client process."""
        with get_session_context() as session:
            process = session.query(ClientProcess).filter(
                ClientProcess.id == process_db_id,
                ClientProcess.client_id == client_id
            ).first()
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")
            update_data = updates.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(process, field, value)
            return {"message": "Process updated successfully"}

    @app.delete("/api/v1/clients/{client_id}/processes/{process_db_id}")
    async def delete_client_process(client_id: int, process_db_id: int):
        """Delete a client process."""
        with get_session_context() as session:
            process = session.query(ClientProcess).filter(
                ClientProcess.id == process_db_id,
                ClientProcess.client_id == client_id
            ).first()
            if not process:
                raise HTTPException(status_code=404, detail="Process not found")
            session.delete(process)
            return {"message": "Process deleted successfully"}

    # ---- Client Risk Endpoints ----

    @app.post("/api/v1/clients/{client_id}/risks")
    async def add_client_risk(client_id: int, risk: RiskCreate):
        """Add a risk to a client portfolio."""
        with get_session_context() as session:
            client = session.query(Client).filter(Client.id == client_id).first()
            if not client:
                raise HTTPException(status_code=404, detail="Client not found")
            existing = session.query(ClientRisk).filter(
                ClientRisk.client_id == client_id,
                ClientRisk.risk_id == risk.risk_id
            ).first()
            if existing:
                existing.risk_name = risk.risk_name
                existing.domain = risk.domain
                existing.category = risk.category
                existing.probability = risk.probability
                existing.is_prioritized = risk.is_prioritized
                existing.notes = risk.notes
                return {"id": existing.id, "message": "Risk updated (already existed)"}
            db_risk = ClientRisk(
                client_id=client_id,
                risk_id=risk.risk_id,
                risk_name=risk.risk_name,
                domain=risk.domain,
                category=risk.category,
                probability=risk.probability,
                is_prioritized=risk.is_prioritized,
                notes=risk.notes
            )
            session.add(db_risk)
            session.flush()
            return {"id": db_risk.id, "message": "Risk added successfully"}

    @app.get("/api/v1/clients/{client_id}/risks")
    async def list_client_risks(client_id: int, prioritized_only: bool = False):
        """Get all risks for a client."""
        with get_session_context() as session:
            query = session.query(ClientRisk).filter(
                ClientRisk.client_id == client_id
            )
            if prioritized_only:
                query = query.filter(ClientRisk.is_prioritized == True)
            risks = query.order_by(ClientRisk.probability.desc()).all()
            return {
                "total": len(risks),
                "risks": [
                    {
                        "id": r.id,
                        "client_id": r.client_id,
                        "risk_id": r.risk_id,
                        "risk_name": r.risk_name,
                        "domain": r.domain,
                        "category": r.category,
                        "probability": r.probability,
                        "is_prioritized": r.is_prioritized,
                        "notes": r.notes,
                        "created_at": r.created_at.isoformat() if r.created_at else None
                    }
                    for r in risks
                ]
            }

    @app.put("/api/v1/clients/{client_id}/risks/{risk_db_id}")
    async def update_client_risk(client_id: int, risk_db_id: int, updates: RiskUpdate):
        """Update a client risk."""
        with get_session_context() as session:
            risk = session.query(ClientRisk).filter(
                ClientRisk.id == risk_db_id,
                ClientRisk.client_id == client_id
            ).first()
            if not risk:
                raise HTTPException(status_code=404, detail="Risk not found")
            update_data = updates.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(risk, field, value)
            return {"message": "Risk updated successfully"}

    # ---- Risk Assessment Endpoints ----

    @app.post("/api/v1/clients/{client_id}/assessments")
    async def save_assessment(client_id: int, assessment: AssessmentCreate):
        """Save or update a risk assessment for a process-risk combination."""
        with get_session_context() as session:
            client = session.query(Client).filter(Client.id == client_id).first()
            if not client:
                raise HTTPException(status_code=404, detail="Client not found")
            existing = session.query(ClientRiskAssessment).filter(
                ClientRiskAssessment.client_id == client_id,
                ClientRiskAssessment.process_id == assessment.process_id,
                ClientRiskAssessment.risk_id == assessment.risk_id
            ).first()
            if existing:
                existing.vulnerability = assessment.vulnerability
                existing.resilience = assessment.resilience
                existing.expected_downtime = assessment.expected_downtime
                existing.notes = assessment.notes
                existing.assessed_at = datetime.utcnow()
                return {"id": existing.id, "message": "Assessment updated"}
            db_assessment = ClientRiskAssessment(
                client_id=client_id,
                process_id=assessment.process_id,
                risk_id=assessment.risk_id,
                vulnerability=assessment.vulnerability,
                resilience=assessment.resilience,
                expected_downtime=assessment.expected_downtime,
                notes=assessment.notes,
                assessed_at=datetime.utcnow()
            )
            session.add(db_assessment)
            session.flush()
            return {"id": db_assessment.id, "message": "Assessment saved"}

    @app.get("/api/v1/clients/{client_id}/assessments")
    async def list_assessments(client_id: int):
        """Get all assessments for a client with process and risk details."""
        with get_session_context() as session:
            assessments = session.query(
                ClientRiskAssessment,
                ClientProcess.process_name,
                ClientProcess.custom_name,
                ClientProcess.criticality_per_day,
                ClientProcess.category.label('process_category'),
                ClientRisk.risk_name,
                ClientRisk.domain,
                ClientRisk.category.label('risk_category'),
                ClientRisk.probability
            ).join(
                ClientProcess, ClientRiskAssessment.process_id == ClientProcess.id
            ).join(
                ClientRisk, ClientRiskAssessment.risk_id == ClientRisk.id
            ).filter(
                ClientRiskAssessment.client_id == client_id
            ).order_by(
                ClientProcess.criticality_per_day.desc(),
                ClientRisk.probability.desc()
            ).all()
            results = []
            for row in assessments:
                a = row[0]
                criticality = row.criticality_per_day or 0
                vuln = a.vulnerability or 0.5
                resil = a.resilience or 0.3
                downtime = a.expected_downtime or 5
                prob = row.probability or 0.5
                exposure = criticality * vuln * (1 - resil) * downtime * prob
                results.append({
                    "id": a.id,
                    "client_id": a.client_id,
                    "process_id": a.process_id,
                    "risk_id": a.risk_id,
                    "vulnerability": a.vulnerability,
                    "resilience": a.resilience,
                    "expected_downtime": a.expected_downtime,
                    "notes": a.notes,
                    "assessed_at": a.assessed_at.isoformat() if a.assessed_at else None,
                    "process_name": row.process_name,
                    "custom_name": row.custom_name,
                    "criticality_per_day": row.criticality_per_day,
                    "process_category": row.process_category,
                    "risk_name": row.risk_name,
                    "domain": row.domain,
                    "risk_category": row.risk_category,
                    "probability": row.probability,
                    "exposure": round(exposure, 2)
                })
            return {
                "total": len(results),
                "assessments": results
            }

    # ---- Exposure Summary ----

    @app.get("/api/v1/clients/{client_id}/exposure-summary")
    async def get_exposure_summary(client_id: int):
        """Get comprehensive risk exposure summary for a client."""
        with get_session_context() as session:
            client = session.query(Client).filter(Client.id == client_id).first()
            if not client:
                raise HTTPException(status_code=404, detail="Client not found")
            assessments = session.query(
                ClientRiskAssessment,
                ClientProcess.process_name,
                ClientProcess.custom_name,
                ClientProcess.criticality_per_day,
                ClientRisk.risk_name,
                ClientRisk.domain,
                ClientRisk.probability
            ).join(
                ClientProcess, ClientRiskAssessment.process_id == ClientProcess.id
            ).join(
                ClientRisk, ClientRiskAssessment.risk_id == ClientRisk.id
            ).filter(
                ClientRiskAssessment.client_id == client_id
            ).all()
            total_exposure = 0
            by_domain = {}
            by_process = {}
            by_risk = {}
            details = []
            for row in assessments:
                a = row[0]
                criticality = row.criticality_per_day or 0
                vuln = a.vulnerability or 0.5
                resil = a.resilience or 0.3
                downtime = a.expected_downtime or 5
                prob = row.probability or 0.5
                exposure = criticality * vuln * (1 - resil) * downtime * prob
                total_exposure += exposure
                domain = row.domain or "Unknown"
                by_domain[domain] = by_domain.get(domain, 0) + exposure
                process_name = row.custom_name or row.process_name
                by_process[process_name] = by_process.get(process_name, 0) + exposure
                risk_name = row.risk_name
                by_risk[risk_name] = by_risk.get(risk_name, 0) + exposure
                details.append({
                    "process_name": process_name,
                    "risk_name": risk_name,
                    "domain": domain,
                    "exposure": round(exposure, 2),
                    "vulnerability": a.vulnerability,
                    "resilience": a.resilience,
                    "expected_downtime": a.expected_downtime,
                    "probability": row.probability,
                    "criticality_per_day": row.criticality_per_day
                })
            return {
                "client_id": client_id,
                "client_name": client.name,
                "currency": client.currency,
                "revenue": client.revenue,
                "total_exposure": round(total_exposure, 2),
                "revenue_at_risk_pct": round((total_exposure / client.revenue * 100), 2) if client.revenue and client.revenue > 0 else None,
                "by_domain": {k: round(v, 2) for k, v in sorted(by_domain.items(), key=lambda x: -x[1])},
                "by_process": {k: round(v, 2) for k, v in sorted(by_process.items(), key=lambda x: -x[1])},
                "by_risk": {k: round(v, 2) for k, v in sorted(by_risk.items(), key=lambda x: -x[1])},
                "assessment_count": len(details),
                "details": details
            }
