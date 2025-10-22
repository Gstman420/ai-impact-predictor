"""
AI Service - Updated to use all 5 models from model_manager
"""
from typing import List, Dict
from models.model_manager import model_manager
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class AIService:
    """AI Service using all 5 HuggingFace models"""

    def __init__(self):
        self.model_manager = model_manager
        print("🤖 AI Service initialized with 5 models!")

    def classify_requirements(self, requirements: List[Dict]) -> List[Dict]:
        """
        Classify each requirement using Model 1: Requirement Classifier
        Returns: List of requirements with AI classification
        """
        classified = []
        
        for req in requirements:
            try:
                # Use Model 1: Classifier
                result = self.model_manager.classify_requirement(req['text'])
                classified.append({
                    **req,
                    'ai_category': result['category'],
                    'confidence': result['confidence'],
                    'category_scores': result['all_probabilities']
                })
                print(f"✅ Classified {req['req_id']}: {result['category']} ({result['confidence']:.2f})")
            except Exception as e:
                print(f"⚠️ Classification error for {req['req_id']}: {e}")
                classified.append({
                    **req,
                    'ai_category': 'Unknown',
                    'confidence': 0.0,
                    'error': str(e)
                })
        
        return classified

    def find_dependencies(self, requirements: List[Dict], threshold: float = 0.6) -> Dict:
        """
        Find dependencies using Model 2: Embedder (Qwen3)
        """
        try:
            print(f"🔗 Finding dependencies with threshold {threshold}...")
            
            # Generate embeddings for all requirements
            texts = [req['text'] for req in requirements]
            embeddings = []
            
            for text in texts:
                embedding = self.model_manager.generate_embedding(text)
                embeddings.append(embedding)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            # Calculate similarities
            dependencies = []
            for i in range(len(requirements)):
                for j in range(i + 1, len(requirements)):
                    similarity = cosine_similarity(
                        embeddings_array[i].reshape(1, -1),
                        embeddings_array[j].reshape(1, -1)
                    )[0][0]
                    
                    if similarity >= threshold:
                        dep_type = 'strong' if similarity > 0.8 else \
                                  'moderate' if similarity > 0.7 else 'weak'
                        
                        dependencies.append({
                            'source_id': requirements[i]['req_id'],
                            'target_id': requirements[j]['req_id'],
                            'similarity_score': round(float(similarity), 3),
                            'dependency_type': dep_type
                        })
            
            print(f"✅ Found {len(dependencies)} dependencies")
            return {
                "dependencies": dependencies,
                "total": len(dependencies),
                "threshold_used": threshold,
                "status": "success"
            }
        except Exception as e:
            print(f"❌ Dependency analysis error: {e}")
            return {
                "dependencies": [],
                "total": 0,
                "error": str(e),
                "status": "failed"
            }

    def analyze_impact(self, changed_req_id: str, requirements: List[Dict], 
                      dependencies: List[Dict]) -> Dict:
        """
        Analyze impact using Model 4: Impact Analyzer (FinBERT)
        """
        try:
            # Find the changed requirement
            changed_req = next((r for r in requirements if r['req_id'] == changed_req_id), None)
            
            if not changed_req:
                return {
                    "error": f"Requirement {changed_req_id} not found",
                    "status": "failed"
                }
            
            # Use Model 4: Impact Analyzer
            print(f"📊 Analyzing impact for {changed_req_id}...")
            impact_result = self.model_manager.analyze_impact(changed_req['text'])
            
            # Find cascading impacts
            impacted = self._analyze_impact_chain(
                changed_req_id, 
                requirements, 
                dependencies,
                max_depth=3
            )
            
            # Calculate overall risk
            if impacted:
                avg_impact = sum(imp['impact_score'] for imp in impacted) / len(impacted)
                risk_level = 'Critical' if avg_impact > 0.7 else \
                            'High' if avg_impact > 0.5 else \
                            'Medium' if avg_impact > 0.3 else 'Low'
            else:
                risk_level = 'Minimal'
            
            print(f"✅ Impact analysis complete: {len(impacted)} requirements affected")
            
            return {
                "changed_requirement": changed_req_id,
                "changed_text": changed_req['text'],
                "impact_sentiment": impact_result['impact'],
                "impact_confidence": impact_result['confidence'],
                "impact_scores": impact_result['scores'],
                "impacted_requirements": impacted[:10],  # Top 10
                "total_impacted": len(impacted),
                "overall_risk_level": risk_level,
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Impact analysis error: {e}")
            return {
                "changed_requirement": changed_req_id,
                "error": str(e),
                "status": "failed"
            }

    def _analyze_impact_chain(self, changed_req_id: str, requirements: List[Dict], 
                            dependencies: List[Dict], max_depth: int = 3) -> List[Dict]:
        """Helper: Analyze cascading impact chain"""
        impacted = []
        visited = set()
        
        def find_impacted(req_id: str, depth: int):
            if depth > max_depth or req_id in visited:
                return
            
            visited.add(req_id)
            
            for dep in dependencies:
                target_id = None
                similarity = dep['similarity_score']
                
                if dep['source_id'] == req_id:
                    target_id = dep['target_id']
                elif dep['target_id'] == req_id:
                    target_id = dep['source_id']
                
                if target_id and target_id not in visited:
                    impact_score = similarity * (1 - depth * 0.2)
                    
                    severity = 'Critical' if impact_score > 0.8 else \
                              'High' if impact_score > 0.6 else \
                              'Medium' if impact_score > 0.4 else 'Low'
                    
                    impacted.append({
                        'impacted_req_id': target_id,
                        'impact_score': round(impact_score, 3),
                        'severity': severity,
                        'depth': depth,
                        'path_from_source': f"{req_id} -> {target_id}"
                    })
                    
                    find_impacted(target_id, depth + 1)
        
        find_impacted(changed_req_id, 1)
        impacted.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return impacted

    def summarize_requirements(self, requirements: List[Dict]) -> Dict:
        """
        Generate summary using Model 3: Summarizer (BART)
        """
        try:
            print(f"📝 Summarizing {len(requirements)} requirements...")
            
            # Combine all requirement texts
            combined_text = " ".join([req['text'] for req in requirements])
            
            # Use Model 3: Summarizer
            summary_result = self.model_manager.summarize_text(
                combined_text,
                max_length=200,
                min_length=50
            )
            
            print(f"✅ Summary generated (compression: {summary_result['compression_ratio']}x)")
            
            return {
                "summary": summary_result['summary'],
                "original_length": summary_result['original_length'],
                "summary_length": summary_result['summary_length'],
                "compression_ratio": summary_result['compression_ratio'],
                "total_requirements": len(requirements),
                "status": "success"
            }
        except Exception as e:
            print(f"❌ Summarization error: {e}")
            return {
                "summary": "",
                "error": str(e),
                "status": "failed"
            }

    def analyze_code(self, code: str) -> Dict:
        """
        Analyze code using Model 5: Code Analyzer (StarCoder2)
        """
        try:
            print(f"💻 Analyzing code snippet...")
            
            # Use Model 5: Code Analyzer
            result = self.model_manager.analyze_code(code, max_length=300)
            
            print(f"✅ Code analysis complete")
            
            return {
                "analysis": result['analysis'],
                "suggestions": result['suggestions'],
                "status": "success"
            }
        except Exception as e:
            print(f"❌ Code analysis error: {e}")
            return {
                "analysis": "Unable to analyze code",
                "suggestions": [],
                "error": str(e),
                "status": "failed"
            }

    def analyze_batch(self, requirements: List[Dict], threshold: float = 0.6) -> Dict:
        """
        Complete analysis: Uses Models 1, 2, and 3
        """
        try:
            # Step 1: Classify (Model 1)
            print("🔍 Step 1/3: Classifying requirements...")
            classified = self.classify_requirements(requirements)
            
            # Step 2: Find dependencies (Model 2)
            print("🔗 Step 2/3: Finding dependencies...")
            dep_result = self.find_dependencies(requirements, threshold)
            
            # Step 3: Summarize (Model 3)
            print("📝 Step 3/3: Generating summary...")
            summary_result = self.summarize_requirements(requirements)
            
            print("✅ Batch analysis complete!")
            
            return {
                "classified_requirements": classified,
                "dependencies": dep_result,
                "summary": summary_result,
                "total_requirements": len(requirements),
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Batch analysis error: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def get_model_status(self) -> Dict:
        """Get status of all 5 models"""
        return self.model_manager.get_model_status()

# Create singleton instance
ai_service = AIService()