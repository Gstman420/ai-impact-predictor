from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from database.connection import get_db
from database import crud
import io

router = APIRouter()

def parse_requirements(contents: bytes, filename: str):
    """Parse requirements from uploaded file"""
    requirements = []
    
    try:
        text = contents.decode('utf-8')
        lines = text.strip().split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append({
                    "req_id": f"REQ-{i:03d}",
                    "text": line,
                    "category": "Functional" if i % 2 == 0 else "Non-Functional",
                    "priority": "High" if i % 3 == 0 else "Medium"
                })
        
        return requirements
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload and parse requirements file"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    allowed_extensions = ['.txt', '.csv', '.md']
    if not any(file.filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Parse requirements
        requirements = parse_requirements(contents, file.filename)
        
        if not requirements:
            raise HTTPException(status_code=400, detail="No valid requirements found in file")
        
        # Save to database
        crud.bulk_create_requirements(db, requirements)
        crud.create_upload_record(db, file.filename, len(requirements))
        
        return {
            "message": "File uploaded and saved to database successfully",
            "filename": file.filename,
            "total_requirements": len(requirements),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/requirements")
async def get_all_requirements(db: Session = Depends(get_db)):
    """Get all requirements from database"""
    requirements = crud.get_all_requirements(db)
    return {
        "total": len(requirements),
        "requirements": [
            {
                "req_id": req.req_id,
                "text": req.text,
                "category": req.category,
                "priority": req.priority,
                "created_at": str(req.created_at)
            }
            for req in requirements
        ]
    }

@router.get("/requirements/search")
async def search_requirements(keyword: str, db: Session = Depends(get_db)):
    """Search requirements by keyword"""
    results = crud.search_requirements(db, keyword)
    return {
        "total": len(results),
        "keyword": keyword,
        "results": [
            {
                "req_id": req.req_id,
                "text": req.text,
                "category": req.category,
                "priority": req.priority
            }
            for req in results
        ]
    }

@router.get("/requirements/filter")
async def filter_requirements(
    category: str = None, 
    priority: str = None, 
    db: Session = Depends(get_db)
):
    """Filter requirements by category and/or priority"""
    results = crud.filter_requirements(db, category, priority)
    return {
        "total": len(results),
        "filters": {"category": category, "priority": priority},
        "results": [
            {
                "req_id": req.req_id,
                "text": req.text,
                "category": req.category,
                "priority": req.priority
            }
            for req in results
        ]
    }

@router.get("/upload-history")
async def get_upload_history(db: Session = Depends(get_db)):
    """Get file upload history"""
    history = crud.get_upload_history(db)
    return {
        "total": len(history),
        "uploads": [
            {
                "id": upload.id,
                "filename": upload.filename,
                "total_requirements": upload.total_requirements,
                "uploaded_at": str(upload.uploaded_at),
                "status": upload.status
            }
            for upload in history
        ]
    }

@router.delete("/requirements")
async def delete_all_requirements(db: Session = Depends(get_db)):
    """Delete all requirements (for testing)"""
    crud.delete_all_requirements(db)
    return {"message": "All requirements deleted successfully"}