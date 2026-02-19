"""
Phase 3 Enhanced Dashboard API Routes
Manages probability trends, alerts, industry profiles, and report scheduling.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, desc, and_

from database.models import (
    RiskEvent, RiskProbability, ProbabilitySnapshot,
    ProbabilityAlert, AlertEvent,
    IndustryProfile, ProfileRiskEvent,
    ReportSchedule, ReportHistory,
    Client
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SnapshotResponse(BaseModel):
    snapshot_date: datetime
    probability_pct: float
    confidence_score: float
    signal: Optional[str]
    momentum: Optional[float]
    trend: Optional[str]


class TrendStatsResponse(BaseModel):
    current_probability: float
    avg_30d: float
    avg_90d: float
    min_30d: float
    max_30d: float
    change_7d: float
    change_30d: float
    trend_direction: str


class MoverResponse(BaseModel):
    event_id: int
    event_name: str
    current_pct: float
    previous_pct: float
    change_pct: float
    direction: str


class TrendSummaryResponse(BaseModel):
    total_events_tracked: int
    latest_snapshot_date: Optional[datetime]
    events_rising: int
    events_falling: int
    events_stable: int
    avg_probability_all: float


class AlertCreate(BaseModel):
    event_id: int
    alert_name: str
    threshold_pct: float
    direction: str  # 'ABOVE', 'BELOW', 'CHANGE'
    severity: str = 'MEDIUM'  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    client_id: Optional[int] = None
    notification_email: Optional[str] = None
    is_active: bool = True


class AlertUpdate(BaseModel):
    alert_name: Optional[str] = None
    threshold_pct: Optional[float] = None
    direction: Optional[str] = None
    severity: Optional[str] = None
    is_active: Optional[bool] = None
    notification_email: Optional[str] = None


class AlertResponse(BaseModel):
    id: int
    event_id: int
    alert_name: str
    threshold_pct: float
    direction: str
    severity: str
    client_id: Optional[int]
    notification_email: Optional[str]
    is_active: bool
    created_at: datetime
    last_triggered_at: Optional[datetime]


class AlertEventResponse(BaseModel):
    id: int
    alert_id: int
    alert_name: str
    event_id: int
    triggered_value: float
    previous_value: Optional[float]
    triggered_at: datetime
    severity: str
    acknowledged: bool


class ProfileCreate(BaseModel):
    industry: str
    profile_name: str
    description: str = ''
    is_template: bool = True
    events: List[Dict[str, Any]] = []  # [{event_id, relevance_score, weight_multiplier, category}]


class ProfileResponse(BaseModel):
    id: int
    industry: str
    profile_name: str
    description: str
    is_template: bool
    created_at: datetime
    event_count: int


class ReportCreate(BaseModel):
    report_name: str
    client_id: Optional[int] = None
    report_type: str = 'WEEKLY'  # 'DAILY', 'WEEKLY', 'MONTHLY'
    report_format: str = 'PDF'  # 'PDF', 'EXCEL', 'JSON'
    recipients: str = ''
    is_active: bool = True
    include_trends: bool = True
    include_alerts: bool = True
    include_recommendations: bool = True


class ReportUpdate(BaseModel):
    report_name: Optional[str] = None
    report_type: Optional[str] = None
    report_format: Optional[str] = None
    recipients: Optional[str] = None
    is_active: Optional[bool] = None
    include_trends: Optional[bool] = None
    include_alerts: Optional[bool] = None
    include_recommendations: Optional[bool] = None


class ReportResponse(BaseModel):
    id: int
    report_name: str
    client_id: Optional[int]
    report_type: str
    report_format: str
    recipients: str
    is_active: bool
    created_at: datetime
    last_generated_at: Optional[datetime]


# ============================================================================
# ROUTE REGISTRATION
# ============================================================================

def register_dashboard_routes(app, get_session_context):
    """Register all Phase 3 Enhanced Dashboard routes."""
    
    router = APIRouter(prefix="/api/v1", tags=["dashboard"])
    
    # ========================================================================
    # PROBABILITY TRENDS (5 endpoints)
    # ========================================================================
    
    @router.get("/trends/movers", response_model=Dict[str, Any])
    def get_probability_movers(
        days: int = Query(7, ge=1, le=365),
        limit: int = Query(20, ge=1, le=100)
    ):
        """Find events with the biggest probability changes over a period."""
        with get_session_context() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get all snapshots within the period
            snapshots = session.query(ProbabilitySnapshot).filter(
                ProbabilitySnapshot.snapshot_date >= cutoff_date
            ).order_by(ProbabilitySnapshot.event_id, desc(ProbabilitySnapshot.snapshot_date)).all()
            
            # Group by event_id and calculate changes
            movers = {}
            for snapshot in snapshots:
                event_id = snapshot.event_id
                if event_id not in movers:
                    movers[event_id] = {'newest': snapshot, 'oldest': snapshot}
                else:
                    movers[event_id]['oldest'] = snapshot
            
            # Calculate changes and prepare response
            mover_list = []
            for event_id, data in movers.items():
                newest = data['newest']
                oldest = data['oldest']
                current_pct = newest.probability_pct
                previous_pct = oldest.probability_pct
                change_pct = current_pct - previous_pct
                direction = 'UP' if change_pct > 0 else 'DOWN' if change_pct < 0 else 'FLAT'
                
                event = session.query(RiskEvent).filter(RiskEvent.event_id == event_id).first()
                event_name = event.event_name if event else f"Event {event_id}"
                
                mover_list.append(MoverResponse(
                    event_id=event_id,
                    event_name=event_name,
                    current_pct=current_pct,
                    previous_pct=previous_pct,
                    change_pct=change_pct,
                    direction=direction
                ))
            
            # Sort by absolute change and limit
            mover_list.sort(key=lambda x: abs(x.change_pct), reverse=True)
            mover_list = mover_list[:limit]
            
            return {
                'total': len(mover_list),
                'movers': mover_list
            }
    

    @router.post("/trends/snapshot", response_model=Dict[str, Any])
    def create_probability_snapshot():
        """Create a snapshot of all current probabilities."""
        with get_session_context() as session:
            probabilities = session.query(RiskProbability).all()
            snapshots_created = 0
            
            for prob in probabilities:
                snapshot = ProbabilitySnapshot(
                    event_id=prob.event_id,
                    probability_pct=prob.probability_pct,
                    confidence_score=prob.confidence_score or 0.5,
                    signal=None,
                    momentum=None,
                    trend=None,
                    snapshot_date=datetime.utcnow()
                )
                session.add(snapshot)
                snapshots_created += 1
            
            session.commit()
            
            return {
                'snapshots_created': snapshots_created,
                'timestamp': datetime.utcnow().isoformat()
            }
    

    @router.get("/trends/summary", response_model=TrendSummaryResponse)
    def get_trend_summary():
        """Get overall trend summary across all events."""
        with get_session_context() as session:
            # Get latest snapshots for each event
            latest_snapshots = session.query(ProbabilitySnapshot).distinct(
                ProbabilitySnapshot.event_id
            ).order_by(
                ProbabilitySnapshot.event_id,
                desc(ProbabilitySnapshot.snapshot_date)
            ).all()
            
            if not latest_snapshots:
                return TrendSummaryResponse(
                    total_events_tracked=0,
                    latest_snapshot_date=None,
                    events_rising=0,
                    events_falling=0,
                    events_stable=0,
                    avg_probability_all=0.0
                )
            
            total_events = len(latest_snapshots)
            latest_date = max(s.snapshot_date for s in latest_snapshots) if latest_snapshots else None
            
            events_rising = sum(1 for s in latest_snapshots if s.trend == 'RISING')
            events_falling = sum(1 for s in latest_snapshots if s.trend == 'FALLING')
            events_stable = sum(1 for s in latest_snapshots if s.trend == 'STABLE')
            avg_probability = sum(s.probability_pct for s in latest_snapshots) / len(latest_snapshots) if latest_snapshots else 0
            
            return TrendSummaryResponse(
                total_events_tracked=total_events,
                latest_snapshot_date=latest_date,
                events_rising=events_rising,
                events_falling=events_falling,
                events_stable=events_stable,
                avg_probability_all=avg_probability
            )
    

    @router.get("/trends/{event_id}", response_model=List[SnapshotResponse])
    def get_probability_trends(
        event_id: int,
        days: int = Query(30, ge=1, le=365)
    ):
        """Get probability trend snapshots for an event over the last N days."""
        with get_session_context() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            snapshots = session.query(ProbabilitySnapshot).filter(
                and_(
                    ProbabilitySnapshot.event_id == event_id,
                    ProbabilitySnapshot.snapshot_date >= cutoff_date
                )
            ).order_by(desc(ProbabilitySnapshot.snapshot_date)).all()
            
            return [
                SnapshotResponse(
                    snapshot_date=s.snapshot_date,
                    probability_pct=s.probability_pct,
                    confidence_score=s.confidence_score,
                    signal=s.signal,
                    momentum=s.momentum,
                    trend=s.trend
                )
                for s in snapshots
            ]
    

    @router.get("/trends/{event_id}/stats", response_model=TrendStatsResponse)
    def get_trend_statistics(event_id: int):
        """Calculate trend statistics for an event."""
        with get_session_context() as session:
            now = datetime.utcnow()
            cutoff_30d = now - timedelta(days=30)
            cutoff_90d = now - timedelta(days=90)
            cutoff_7d = now - timedelta(days=7)
            
            # Get all snapshots for calculations
            snapshots_30d = session.query(ProbabilitySnapshot).filter(
                and_(
                    ProbabilitySnapshot.event_id == event_id,
                    ProbabilitySnapshot.snapshot_date >= cutoff_30d
                )
            ).order_by(desc(ProbabilitySnapshot.snapshot_date)).all()
            
            snapshots_90d = session.query(ProbabilitySnapshot).filter(
                and_(
                    ProbabilitySnapshot.event_id == event_id,
                    ProbabilitySnapshot.snapshot_date >= cutoff_90d
                )
            ).all()
            
            snapshots_7d = session.query(ProbabilitySnapshot).filter(
                and_(
                    ProbabilitySnapshot.event_id == event_id,
                    ProbabilitySnapshot.snapshot_date >= cutoff_7d
                )
            ).order_by(desc(ProbabilitySnapshot.snapshot_date)).all()
            
            if not snapshots_30d:
                raise HTTPException(status_code=404, detail="No trend data found for this event")
            
            current = snapshots_30d[0].probability_pct
            avg_30d = sum(s.probability_pct for s in snapshots_30d) / len(snapshots_30d) if snapshots_30d else 0
            avg_90d = sum(s.probability_pct for s in snapshots_90d) / len(snapshots_90d) if snapshots_90d else 0
            min_30d = min(s.probability_pct for s in snapshots_30d) if snapshots_30d else 0
            max_30d = max(s.probability_pct for s in snapshots_30d) if snapshots_30d else 0
            
            change_7d = 0
            change_30d = 0
            
            if len(snapshots_7d) > 1:
                change_7d = snapshots_7d[0].probability_pct - snapshots_7d[-1].probability_pct
            
            if len(snapshots_30d) > 1:
                change_30d = snapshots_30d[0].probability_pct - snapshots_30d[-1].probability_pct
            
            # Determine trend direction
            if change_30d > 2:
                trend_direction = 'RISING'
            elif change_30d < -2:
                trend_direction = 'FALLING'
            else:
                trend_direction = 'STABLE'
            
            return TrendStatsResponse(
                current_probability=current,
                avg_30d=avg_30d,
                avg_90d=avg_90d,
                min_30d=min_30d,
                max_30d=max_30d,
                change_7d=change_7d,
                change_30d=change_30d,
                trend_direction=trend_direction
            )
    
    # ========================================================================
    # ALERTS (7 endpoints)
    # ========================================================================
    
    @router.post("/alerts", response_model=Dict[str, Any])
    def create_alert(alert_data: AlertCreate):
        """Create a new probability alert."""
        with get_session_context() as session:
            alert = ProbabilityAlert(
                event_id=alert_data.event_id,
                alert_name=alert_data.alert_name,
                threshold_pct=alert_data.threshold_pct,
                direction=alert_data.direction,
                severity=alert_data.severity,
                client_id=alert_data.client_id,
                notification_email=alert_data.notification_email,
                is_active=alert_data.is_active,
                created_at=datetime.utcnow()
            )
            session.add(alert)
            session.commit()
            
            return {
                'id': alert.id,
                'message': f'Alert "{alert_data.alert_name}" created successfully'
            }
    
    @router.get("/alerts", response_model=Dict[str, Any])
    def list_alerts(
        client_id: Optional[int] = None,
        active_only: bool = Query(True)
    ):
        """List all alerts, optionally filtered."""
        with get_session_context() as session:
            query = session.query(ProbabilityAlert)
            
            if active_only:
                query = query.filter(ProbabilityAlert.is_active == True)
            
            if client_id is not None:
                query = query.filter(ProbabilityAlert.client_id == client_id)
            
            alerts = query.order_by(desc(ProbabilityAlert.created_at)).all()
            
            return {
                'total': len(alerts),
                'alerts': [
                    AlertResponse(
                        id=a.id,
                        event_id=a.event_id,
                        alert_name=a.alert_name,
                        threshold_pct=a.threshold_pct,
                        direction=a.direction,
                        severity=a.severity,
                        client_id=a.client_id,
                        notification_email=a.notification_email,
                        is_active=a.is_active,
                        created_at=a.created_at,
                        last_triggered_at=a.last_triggered_at
                    )
                    for a in alerts
                ]
            }
    
    @router.get("/alerts/{alert_id}", response_model=Dict[str, Any])
    def get_alert_details(alert_id: int):
        """Get alert with recent trigger history."""
        with get_session_context() as session:
            alert = session.query(ProbabilityAlert).filter(ProbabilityAlert.id == alert_id).first()
            if not alert:
                raise HTTPException(status_code=404, detail="Alert not found")
            
            events = session.query(AlertEvent).filter(
                AlertEvent.alert_id == alert_id
            ).order_by(desc(AlertEvent.triggered_at)).limit(10).all()
            
            return {
                'alert': AlertResponse(
                    id=alert.id,
                    event_id=alert.event_id,
                    alert_name=alert.alert_name,
                    threshold_pct=alert.threshold_pct,
                    direction=alert.direction,
                    severity=alert.severity,
                    client_id=alert.client_id,
                    notification_email=alert.notification_email,
                    is_active=alert.is_active,
                    created_at=alert.created_at,
                    last_triggered_at=alert.last_triggered_at
                ),
                'trigger_history': [
                    {
                        'id': e.id,
                        'triggered_value': e.triggered_value,
                        'previous_value': e.previous_value,
                        'triggered_at': e.triggered_at,
                        'acknowledged': e.acknowledged
                    }
                    for e in events
                ]
            }
    
    @router.put("/alerts/{alert_id}", response_model=Dict[str, Any])
    def update_alert(alert_id: int, alert_data: AlertUpdate):
        """Update an alert."""
        with get_session_context() as session:
            alert = session.query(ProbabilityAlert).filter(ProbabilityAlert.id == alert_id).first()
            if not alert:
                raise HTTPException(status_code=404, detail="Alert not found")
            
            if alert_data.alert_name is not None:
                alert.alert_name = alert_data.alert_name
            if alert_data.threshold_pct is not None:
                alert.threshold_pct = alert_data.threshold_pct
            if alert_data.direction is not None:
                alert.direction = alert_data.direction
            if alert_data.severity is not None:
                alert.severity = alert_data.severity
            if alert_data.is_active is not None:
                alert.is_active = alert_data.is_active
            if alert_data.notification_email is not None:
                alert.notification_email = alert_data.notification_email
            
            session.commit()
            
            return {'message': 'Alert updated successfully'}
    
    @router.delete("/alerts/{alert_id}", response_model=Dict[str, Any])
    def delete_alert(alert_id: int):
        """Delete an alert and cascade AlertEvents."""
        with get_session_context() as session:
            alert = session.query(ProbabilityAlert).filter(ProbabilityAlert.id == alert_id).first()
            if not alert:
                raise HTTPException(status_code=404, detail="Alert not found")
            
            session.query(AlertEvent).filter(AlertEvent.alert_id == alert_id).delete()
            session.delete(alert)
            session.commit()
            
            return {'message': 'Alert deleted successfully'}
    
    @router.post("/alerts/check", response_model=Dict[str, Any])
    def check_all_alerts():
        """Check all active alerts against current probabilities."""
        with get_session_context() as session:
            active_alerts = session.query(ProbabilityAlert).filter(
                ProbabilityAlert.is_active == True
            ).all()
            
            alerts_checked = len(active_alerts)
            alerts_triggered = 0
            triggered_list = []
            
            for alert in active_alerts:
                current_prob = session.query(RiskProbability).filter(
                    RiskProbability.event_id == alert.event_id
                ).order_by(desc(RiskProbability.calculation_date)).first()
                
                if not current_prob:
                    continue
                
                triggered = False
                
                if alert.direction == 'ABOVE':
                    triggered = current_prob.probability_pct > alert.threshold_pct
                elif alert.direction == 'BELOW':
                    triggered = current_prob.probability_pct < alert.threshold_pct
                elif alert.direction == 'CHANGE':
                    last_snapshot = session.query(ProbabilitySnapshot).filter(
                        ProbabilitySnapshot.event_id == alert.event_id
                    ).order_by(desc(ProbabilitySnapshot.snapshot_date)).offset(1).first()
                    
                    if last_snapshot:
                        change = abs(current_prob.probability_pct - last_snapshot.probability_pct)
                        triggered = change > alert.threshold_pct
                
                if triggered:
                    event_alert = AlertEvent(
                        alert_id=alert.id,
                        triggered_value=current_prob.probability_pct,
                        previous_value=None,
                        triggered_at=datetime.utcnow(),
                        acknowledged=False
                    )
                    session.add(event_alert)
                    alert.last_triggered_at = datetime.utcnow()
                    alerts_triggered += 1
                    
                    triggered_list.append({
                        'alert_id': alert.id,
                        'event_id': alert.event_id,
                        'value': current_prob.probability_pct,
                        'threshold': alert.threshold_pct
                    })
            
            session.commit()
            
            return {
                'alerts_checked': alerts_checked,
                'alerts_triggered': alerts_triggered,
                'triggered': triggered_list
            }
    
    @router.get("/alerts/triggered", response_model=Dict[str, Any])
    def get_triggered_alerts(
        days: int = Query(7, ge=1, le=365),
        acknowledged: Optional[bool] = None
    ):
        """Get recent AlertEvents."""
        with get_session_context() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = session.query(AlertEvent).filter(
                AlertEvent.triggered_at >= cutoff_date
            )
            
            if acknowledged is not None:
                query = query.filter(AlertEvent.acknowledged == acknowledged)
            
            events = query.order_by(desc(AlertEvent.triggered_at)).all()
            
            result_events = []
            for event in events:
                alert = session.query(ProbabilityAlert).filter(
                    ProbabilityAlert.id == event.alert_id
                ).first()
                
                if alert:
                    result_events.append(AlertEventResponse(
                        id=event.id,
                        alert_id=event.alert_id,
                        alert_name=alert.alert_name,
                        event_id=alert.event_id,
                        triggered_value=event.triggered_value,
                        previous_value=event.previous_value,
                        triggered_at=event.triggered_at,
                        severity=alert.severity,
                        acknowledged=event.acknowledged
                    ))
            
            return {
                'total': len(result_events),
                'events': result_events
            }
    
    # ========================================================================
    # INDUSTRY PROFILES (5 endpoints)
    # ========================================================================
    
    @router.get("/profiles", response_model=Dict[str, Any])
    def list_profiles():
        """List all industry profiles."""
        with get_session_context() as session:
            profiles = session.query(IndustryProfile).order_by(IndustryProfile.industry).all()
            
            result = []
            for profile in profiles:
                event_count = session.query(ProfileRiskEvent).filter(
                    ProfileRiskEvent.profile_id == profile.id
                ).count()
                
                result.append(ProfileResponse(
                    id=profile.id,
                    industry=profile.industry,
                    profile_name=profile.profile_name,
                    description=profile.description or '',
                    is_template=profile.is_template,
                    created_at=profile.created_at,
                    event_count=event_count
                ))
            
            return {
                'total': len(result),
                'profiles': result
            }
    
    @router.get("/profiles/{profile_id}", response_model=Dict[str, Any])
    def get_profile_details(profile_id: int):
        """Get a profile with all its risk events."""
        with get_session_context() as session:
            profile = session.query(IndustryProfile).filter(
                IndustryProfile.id == profile_id
            ).first()
            
            if not profile:
                raise HTTPException(status_code=404, detail="Profile not found")
            
            profile_events = session.query(ProfileRiskEvent).filter(
                ProfileRiskEvent.profile_id == profile_id
            ).all()
            
            events_list = []
            for pe in profile_events:
                event = session.query(RiskEvent).filter(RiskEvent.event_id == pe.event_id).first()
                if event:
                    events_list.append({
                        'event_id': event.event_id,
                        'event_name': event.event_name,
                        'relevance_score': pe.relevance_score,
                        'weight_multiplier': pe.weight_multiplier,
                        'category': pe.category or ''
                    })
            
            return {
                'profile': ProfileResponse(
                    id=profile.id,
                    industry=profile.industry,
                    profile_name=profile.profile_name,
                    description=profile.description or '',
                    is_template=profile.is_template,
                    created_at=profile.created_at,
                    event_count=len(events_list)
                ),
                'events': events_list
            }
    
    @router.post("/profiles", response_model=Dict[str, Any])
    def create_profile(profile_data: ProfileCreate):
        """Create an industry profile with associated events."""
        with get_session_context() as session:
            profile = IndustryProfile(
                industry=profile_data.industry,
                profile_name=profile_data.profile_name,
                description=profile_data.description,
                is_template=profile_data.is_template,
                created_at=datetime.utcnow()
            )
            session.add(profile)
            session.flush()  # Get the profile ID
            
            events_added = 0
            for event_data in profile_data.events:
                profile_event = ProfileRiskEvent(
                    profile_id=profile.id,
                    event_id=event_data.get('event_id'),
                    relevance_score=event_data.get('relevance_score', 1.0),
                    weight_multiplier=event_data.get('weight_multiplier', 1.0),
                    category=event_data.get('category', '')
                )
                session.add(profile_event)
                events_added += 1
            
            session.commit()
            
            return {
                'id': profile.id,
                'events_added': events_added,
                'message': f'Profile "{profile_data.profile_name}" created successfully'
            }
    
    @router.get("/profiles/industry/{industry}", response_model=Dict[str, Any])
    def get_profile_by_industry(industry: str):
        """Find profile by industry name (case-insensitive)."""
        with get_session_context() as session:
            profile = session.query(IndustryProfile).filter(
                func.lower(IndustryProfile.industry) == func.lower(industry)
            ).first()
            
            if not profile:
                raise HTTPException(status_code=404, detail="Profile not found for this industry")
            
            profile_events = session.query(ProfileRiskEvent).filter(
                ProfileRiskEvent.profile_id == profile.id
            ).all()
            
            events_list = []
            for pe in profile_events:
                event = session.query(RiskEvent).filter(RiskEvent.event_id == pe.event_id).first()
                if event:
                    events_list.append({
                        'event_id': event.event_id,
                        'event_name': event.event_name,
                        'relevance_score': pe.relevance_score,
                        'weight_multiplier': pe.weight_multiplier,
                        'category': pe.category or ''
                    })
            
            return {
                'profile': ProfileResponse(
                    id=profile.id,
                    industry=profile.industry,
                    profile_name=profile.profile_name,
                    description=profile.description or '',
                    is_template=profile.is_template,
                    created_at=profile.created_at,
                    event_count=len(events_list)
                ),
                'events': events_list
            }
    
    @router.post("/profiles/{profile_id}/apply/{client_id}", response_model=Dict[str, Any])
    def apply_profile_to_client(profile_id: int, client_id: int):
        """Apply an industry profile to a client."""
        with get_session_context() as session:
            profile = session.query(IndustryProfile).filter(
                IndustryProfile.id == profile_id
            ).first()
            
            if not profile:
                raise HTTPException(status_code=404, detail="Profile not found")
            
            client = session.query(Client).filter(Client.id == client_id).first()
            if not client:
                raise HTTPException(status_code=404, detail="Client not found")
            
            profile_events = session.query(ProfileRiskEvent).filter(
                ProfileRiskEvent.profile_id == profile_id
            ).all()
            
            risks_added = 0
            risks_updated = 0
            
            for pe in profile_events:
                event = session.query(RiskEvent).filter(RiskEvent.id == pe.event_id).first()
                if not event:
                    continue
                
                # Check if a probability already exists for this event
                existing_prob = session.query(RiskProbability).filter(
                    RiskProbability.event_id == pe.event_id
                ).first()

                if existing_prob:
                    risks_updated += 1
                else:
                    # Create new RiskProbability with relevance_score as initial probability
                    new_prob = RiskProbability(
                        event_id=pe.event_id,
                        probability_pct=pe.relevance_score * 100,
                        confidence_score=0.5,
                        calculation_method='profile_application',
                        calculation_date=datetime.utcnow()
                    )
                    session.add(new_prob)
                    risks_added += 1
            
            session.commit()
            
            return {
                'risks_added': risks_added,
                'risks_updated': risks_updated,
                'message': f'Profile applied to client {client_id} successfully'
            }
    
    # ========================================================================
    # REPORT MANAGEMENT (5 endpoints)
    # ========================================================================
    
    @router.post("/reports", response_model=Dict[str, Any])
    def create_report(report_data: ReportCreate):
        """Create a new report schedule."""
        with get_session_context() as session:
            report = ReportSchedule(
                report_name=report_data.report_name,
                client_id=report_data.client_id,
                report_type=report_data.report_type,
                report_format=report_data.report_format,
                recipients=report_data.recipients,
                is_active=report_data.is_active,
                include_trends=report_data.include_trends,
                include_alerts=report_data.include_alerts,
                include_recommendations=report_data.include_recommendations,
                created_at=datetime.utcnow()
            )
            session.add(report)
            session.commit()
            
            return {
                'id': report.id,
                'message': f'Report "{report_data.report_name}" created successfully'
            }
    
    @router.get("/reports", response_model=Dict[str, Any])
    def list_reports(
        client_id: Optional[int] = None,
        active_only: bool = Query(False)
    ):
        """List all report schedules, optionally filtered."""
        with get_session_context() as session:
            query = session.query(ReportSchedule)
            
            if active_only:
                query = query.filter(ReportSchedule.is_active == True)
            
            if client_id is not None:
                query = query.filter(ReportSchedule.client_id == client_id)
            
            reports = query.order_by(desc(ReportSchedule.created_at)).all()
            
            return {
                'total': len(reports),
                'reports': [
                    ReportResponse(
                        id=r.id,
                        report_name=r.report_name,
                        client_id=r.client_id,
                        report_type=r.report_type,
                        report_format=r.report_format,
                        recipients=r.recipients,
                        is_active=r.is_active,
                        created_at=r.created_at,
                        last_generated_at=r.last_generated_at
                    )
                    for r in reports
                ]
            }
    
    @router.get("/reports/{report_id}", response_model=Dict[str, Any])
    def get_report_details(report_id: int):
        """Get report schedule with generation history."""
        with get_session_context() as session:
            report = session.query(ReportSchedule).filter(
                ReportSchedule.id == report_id
            ).first()
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            history = session.query(ReportHistory).filter(
                ReportHistory.report_id == report_id
            ).order_by(desc(ReportHistory.generated_at)).limit(10).all()
            
            return {
                'report': ReportResponse(
                    id=report.id,
                    report_name=report.report_name,
                    client_id=report.client_id,
                    report_type=report.report_type,
                    report_format=report.report_format,
                    recipients=report.recipients,
                    is_active=report.is_active,
                    created_at=report.created_at,
                    last_generated_at=report.last_generated_at
                ),
                'history': [
                    {
                        'id': h.id,
                        'generated_at': h.generated_at,
                        'status': h.status,
                        'file_path': h.file_path
                    }
                    for h in history
                ]
            }
    
    @router.put("/reports/{report_id}", response_model=Dict[str, Any])
    def update_report(report_id: int, report_data: ReportUpdate):
        """Update a report schedule."""
        with get_session_context() as session:
            report = session.query(ReportSchedule).filter(
                ReportSchedule.id == report_id
            ).first()
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            if report_data.report_name is not None:
                report.report_name = report_data.report_name
            if report_data.report_type is not None:
                report.report_type = report_data.report_type
            if report_data.report_format is not None:
                report.report_format = report_data.report_format
            if report_data.recipients is not None:
                report.recipients = report_data.recipients
            if report_data.is_active is not None:
                report.is_active = report_data.is_active
            if report_data.include_trends is not None:
                report.include_trends = report_data.include_trends
            if report_data.include_alerts is not None:
                report.include_alerts = report_data.include_alerts
            if report_data.include_recommendations is not None:
                report.include_recommendations = report_data.include_recommendations
            
            session.commit()
            
            return {'message': 'Report updated successfully'}
    
    @router.post("/reports/{report_id}/generate", response_model=Dict[str, Any])
    def generate_report_now(report_id: int):
        """Generate a report on-demand."""
        with get_session_context() as session:
            report = session.query(ReportSchedule).filter(
                ReportSchedule.id == report_id
            ).first()
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            # Gather report data
            report_data = {
                'summary': {
                    'report_name': report.report_name,
                    'report_type': report.report_type,
                    'client_id': report.client_id,
                    'generated_at': datetime.utcnow().isoformat()
                },
                'trends': [],
                'alerts': [],
                'recommendations': []
            }
            
            # Include trends if requested
            if report.include_trends:
                snapshots = session.query(ProbabilitySnapshot).order_by(
                    desc(ProbabilitySnapshot.snapshot_date)
                ).limit(20).all()
                
                report_data['trends'] = [
                    {
                        'event_id': s.event_id,
                        'probability_pct': s.probability_pct,
                        'snapshot_date': s.snapshot_date.isoformat()
                    }
                    for s in snapshots
                ]
            
            # Include alerts if requested
            if report.include_alerts:
                alerts = session.query(AlertEvent).order_by(
                    desc(AlertEvent.triggered_at)
                ).limit(20).all()
                
                report_data['alerts'] = [
                    {
                        'alert_id': a.alert_id,
                        'triggered_value': a.triggered_value,
                        'triggered_at': a.triggered_at.isoformat(),
                        'acknowledged': a.acknowledged
                    }
                    for a in alerts
                ]
            
            # Create ReportHistory entry
            history_entry = ReportHistory(
                report_id=report_id,
                generated_at=datetime.utcnow(),
                status='GENERATED',
                file_path=None
            )
            session.add(history_entry)
            report.last_generated_at = datetime.utcnow()
            session.commit()
            
            return {
                'report_id': report_id,
                'generated_at': datetime.utcnow().isoformat(),
                'data': report_data
            }
    
    app.include_router(router)
