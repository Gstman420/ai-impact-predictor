from sqlalchemy.orm import Session
from database.models import Requirement, Dependency, ImpactAnalysis, UploadHistory
from typing import List, Dict, Optional
from datetime import datetime

# ============== REQUIREMENTS ==============

def create_requirement(db: Session, req_data: Dict):
    """Create a new requirement"""
    req = Requirement(
        req_id=req_data["req_id"],
        text=req_data["text"],
        category=req_data.get("category", "Not Specified"),
        priority=req_data.get("priority", "Medium")
    )
    db.add(req)
    db.commit()
    db.refresh(req)
    return req

def get_requirement(db: Session, req_id: str):
    """Get a single requirement by ID"""
    return db.query(Requirement).filter(Requirement.req_id == req_id).first()

def get_all_requirements(db: Session, skip: int = 0, limit: int = 100):
    """Get all requirements with pagination"""
    return db.query(Requirement).offset(skip).limit(limit).all()

def delete_all_requirements(db: Session):
    """Delete all requirements"""
    db.query(Requirement).delete()
    db.commit()

def search_requirements(db: Session, keyword: str):
    """Search requirements by keyword"""
    return db.query(Requirement).filter(
        Requirement.text.contains(keyword)
    ).all()

def filter_requirements(db: Session, category: Optional[str] = None, priority: Optional[str] = None):
    """Filter requirements by category and/or priority"""
    query = db.query(Requirement)
    if category:
        query = query.filter(Requirement.category == category)
    if priority:
        query = query.filter(Requirement.priority == priority)
    return query.all()

def bulk_create_requirements(db: Session, requirements: List[Dict]):
    """Create multiple requirements at once"""
    for req_data in requirements:
        req = Requirement(
            req_id=req_data["req_id"],
            text=req_data["text"],
            category=req_data.get("category", "Not Specified"),
            priority=req_data.get("priority", "Medium")
        )
        db.add(req)
    db.commit()

# ============== DEPENDENCIES ==============

def create_dependency(db: Session, dep_data: Dict):
    """Create a new dependency"""
    dep = Dependency(
        source_id=dep_data["source_id"],
        target_id=dep_data["target_id"],
        similarity_score=dep_data["similarity_score"],
        dependency_type=dep_data.get("dependency_type", "related")
    )
    db.add(dep)
    db.commit()
    db.refresh(dep)
    return dep

def get_dependencies(db: Session, req_id: Optional[str] = None):
    """Get all dependencies or dependencies for a specific requirement"""
    query = db.query(Dependency)
    if req_id:
        query = query.filter(
            (Dependency.source_id == req_id) | (Dependency.target_id == req_id)
        )
    return query.all()

def bulk_create_dependencies(db: Session, dependencies: List[Dict]):
    """Create multiple dependencies at once"""
    for dep_data in dependencies:
        dep = Dependency(
            source_id=dep_data["source_id"],
            target_id=dep_data["target_id"],
            similarity_score=dep_data["similarity_score"],
            dependency_type=dep_data.get("dependency_type", "related")
        )
        db.add(dep)
    db.commit()

# ============== IMPACT ANALYSIS ==============

def create_impact_analysis(db: Session, impact_data: Dict):
    """Create a new impact analysis record"""
    impact = ImpactAnalysis(
        changed_req_id=impact_data["changed_req_id"],
        impacted_req_id=impact_data["impacted_req_id"],
        impact_score=impact_data["impact_score"],
        severity=impact_data.get("severity", "medium"),
        risk_level=impact_data.get("risk_level", "moderate")
    )
    db.add(impact)
    db.commit()
    db.refresh(impact)
    return impact

def get_impact_history(db: Session, req_id: Optional[str] = None, limit: int = 50):
    """Get impact analysis history"""
    query = db.query(ImpactAnalysis)
    if req_id:
        query = query.filter(ImpactAnalysis.changed_req_id == req_id)
    return query.order_by(ImpactAnalysis.analyzed_at.desc()).limit(limit).all()

def bulk_create_impacts(db: Session, impacts: List[Dict]):
    """Create multiple impact analyses at once"""
    for impact_data in impacts:
        impact = ImpactAnalysis(
            changed_req_id=impact_data["changed_req_id"],
            impacted_req_id=impact_data["impacted_req_id"],
            impact_score=impact_data["impact_score"],
            severity=impact_data.get("severity", "medium"),
            risk_level=impact_data.get("risk_level", "moderate")
        )
        db.add(impact)
    db.commit()

# ============== UPLOAD HISTORY ==============

def create_upload_record(db: Session, filename: str, total_reqs: int):
    """Record a file upload"""
    upload = UploadHistory(
        filename=filename,
        total_requirements=total_reqs
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload

def get_upload_history(db: Session, limit: int = 20):
    """Get upload history"""
    return db.query(UploadHistory).order_by(
        UploadHistory.uploaded_at.desc()
    ).limit(limit).all()