from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from database.connection import get_db
from database import crud
from services.ai_service import ai_service
from services.code_impact_service import code_impact_service
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from models.model_manager import model_manager

router = APIRouter()

@router.post("/analysis/impact-predictor")
async def full_impact_predictor(request: Request):
    """
    Intelligent Code Impact Predictor
    - Analyzes codebase structure
    - Predicts affected files
    - Suggests specific changes
    """
    try:
        data = await request.json()
        text = data.get("text", "").strip()
        files = data.get("files", [])  # Array of {name, content, type}
        
        if not text and not files:
            raise HTTPException(status_code=400, detail="No text or files provided")
        
        # Process uploaded code files
        code_files = []
        csv_content = []
        
        if files:
            for file in files:
                file_name = file.get("name", "")
                file_content = file.get("content", "")
                
                # Separate CSV and code files
                if file_name.endswith('.csv'):
                    csv_content.append({
                        "name": file_name,
                        "content": file_content
                    })
                elif any(file_name.endswith(ext) for ext in ['.py', '.js', '.java', '.cpp', '.c', '.ts', '.jsx', '.tsx', '.go', '.rs', '.sql']):
                    code_files.append({
                        "name": file_name,
                        "content": file_content,
                        "type": file_name.split('.')[-1]
                    })
        
        # Use intelligent code impact analysis if code files are provided
        if code_files:
            print(f"🎯 Running intelligent code impact analysis on {len(code_files)} files...")
            
            # Use the new intelligent service
            impact_result = code_impact_service.analyze_codebase_impact(
                requirement=text,
                codebase_files=code_files
            )
            
            if impact_result.get('status') == 'failed':
                raise HTTPException(status_code=500, detail=impact_result.get('error', 'Analysis failed'))
            
            # Format the response
            result = {
                "input": text,
                "files_processed": {
                    "code_files": len(code_files),
                    "csv_files": len(csv_content),
                    "total": len(files)
                },
                "classification": impact_result.get("classification", "Unknown"),
                "classification_confidence": impact_result.get("classification_confidence", 0),
                "summary": f"Requirement: {text}. Analysis complete for {len(code_files)} file(s).",
                "impact": impact_result.get("impact", "Neutral"),
                "impact_confidence": impact_result.get("impact_confidence", 0),
                "affected_files": impact_result.get("affected_files", []),
                "suggested_changes": impact_result.get("suggested_changes", []),
                "code_analysis": impact_result.get("reasoning", "Analysis complete"),
                "ai_reasoning": impact_result.get("reasoning", "AI has analyzed the codebase structure"),
            }
            
            return {"status": "success", "results": result}
        
        else:
            # Fallback: Simple text analysis without code files
            print("📝 Running simple text analysis (no code files provided)...")
            
            combined_text = text
            if csv_content:
                combined_text += "\n" + csv_content[0]['content'][:500]
            
            # Run basic models
            classification = model_manager.classify_requirement(combined_text)
            summary = model_manager.summarize_text(combined_text)
            impact = model_manager.analyze_impact(combined_text)
            embeddings = model_manager.get_embeddings([combined_text])
            code_analysis = model_manager.analyze_code(combined_text)
            
            # Build reasoning
            ai_reasoning = (
                f"This requirement is classified as {classification.get('classification', 'Unknown')} "
                f"({round(classification.get('confidence', 0)*100, 2)}% confidence). "
                f"The sentiment indicates a {impact.get('impact', 'Neutral').lower()} impact. "
                f"Summary: '{summary}'. "
            )
            
            result = {
                "input": text,
                "files_processed": {
                    "code_files": 0,
                    "csv_files": len(csv_content),
                    "total": len(files)
                },
                "classification": classification.get("classification", "Unknown"),
                "classification_confidence": classification.get("confidence", 0),
                "summary": summary,
                "impact": impact.get("impact", "Neutral"),
                "impact_confidence": impact.get("confidence", 0),
                "embedding_vector_size": len(embeddings[0]) if len(embeddings) > 0 else 0,
                "code_analysis": code_analysis[:300] if code_analysis else "No code analysis available",
                "ai_reasoning": ai_reasoning,
                "affected_files": [],
                "suggested_changes": ["Upload code files for detailed impact analysis"]
            }
            
            return {"status": "success", "results": result}
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")