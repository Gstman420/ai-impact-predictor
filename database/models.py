from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Requirement(Base):
    """Requirements table"""
    __tablename__ = "requirements"
    
    id = Column(Integer, primary_key=True, index=True)
    req_id = Column(String(50), unique=True, index=True, nullable=False)
    text = Column(Text, nullable=False)
    category = Column(String(100))
    priority = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dependencies = relationship("Dependency", foreign_keys="Dependency.source_id", back_populates="source")
    impacts = relationship("ImpactAnalysis", back_populates="requirement")

class Dependency(Base):
    """Dependencies between requirements"""
    __tablename__ = "dependencies"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String(50), ForeignKey("requirements.req_id"), nullable=False)
    target_id = Column(String(50), ForeignKey("requirements.req_id"), nullable=False)
    similarity_score = Column(Float, nullable=False)
    dependency_type = Column(String(50))  # 'depends' or 'related'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    source = relationship("Requirement", foreign_keys=[source_id])

class ImpactAnalysis(Base):
    """Impact analysis history"""
    __tablename__ = "impact_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    changed_req_id = Column(String(50), ForeignKey("requirements.req_id"), nullable=False)
    impacted_req_id = Column(String(50), nullable=False)
    impact_score = Column(Float, nullable=False)
    severity = Column(String(50))  # 'high', 'medium', 'low'
    risk_level = Column(String(50))
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    requirement = relationship("Requirement", back_populates="impacts")

class UploadHistory(Base):
    """Track file uploads"""
    __tablename__ = "upload_history"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    total_requirements = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="success")