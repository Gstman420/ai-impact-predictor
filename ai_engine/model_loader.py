
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification,
    pipeline
)
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ModelManager:
    """Manages all AI models for requirements analysis"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        
        # Model names - UPDATED to use your 8B model
        self.models = {
            'classifier': 'rajinikarcg/RequirementClassifier',
            'embedder': 'Qwen/Qwen3-Embedding-8B',
            'summarizer': 'marklicata/M365_h2_Text_Processing_and_Summarization',
            'financial': 'nusret35/roberta-financial-news-impact-analysis'
        }
        
        # Lazy loading - models load when first used
        self._classifier = None
        self._classifier_tokenizer = None
        self._embedder = None
        self._embedder_tokenizer = None
        self._summarizer = None
        self._financial = None
        self._financial_tokenizer = None
        
    def load_classifier(self):
        """Load requirement classifier model"""
        if self._classifier is None:
            print(f"📦 Loading classifier: {self.models['classifier']}")
            self._classifier_tokenizer = AutoTokenizer.from_pretrained(
                self.models['classifier'],
                local_files_only=True
            )
            self._classifier = AutoModelForSequenceClassification.from_pretrained(
                self.models['classifier'],
                local_files_only=True
            ).to(self.device)
            print("✅ Classifier loaded!")
        return self._classifier, self._classifier_tokenizer
    
    def load_embedder(self):
        """Load embedding model"""
        if self._embedder is None:
            print(f"📦 Loading embedder: {self.models['embedder']}")
            self._embedder_tokenizer = AutoTokenizer.from_pretrained(
                self.models['embedder'],
                local_files_only=True
            )
            self._embedder = AutoModel.from_pretrained(
                self.models['embedder'],
                local_files_only=True
            ).to(self.device)
            print("✅ Embedder loaded!")
        return self._embedder, self._embedder_tokenizer
    
    def load_summarizer(self):
        """Load summarization model"""
        if self._summarizer is None:
            print(f"📦 Loading summarizer: {self.models['summarizer']}")
            self._summarizer = pipeline(
                "summarization",
                model=self.models['summarizer'],
                local_files_only=True,
                device=0 if self.device == "cuda" else -1
            )
            print("✅ Summarizer loaded!")
        return self._summarizer
    
    def load_financial_analyzer(self):
        """Load financial impact analyzer"""
        if self._financial is None:
            print(f"📦 Loading financial analyzer: {self.models['financial']}")
            self._financial_tokenizer = AutoTokenizer.from_pretrained(
                self.models['financial'],
                local_files_only=True
            )
            self._financial = AutoModelForSequenceClassification.from_pretrained(
                self.models['financial'],
                local_files_only=True
            ).to(self.device)
            print("✅ Financial analyzer loaded!")
        return self._financial, self._financial_tokenizer
    
    def classify_requirement(self, text: str) -> Dict:
        """Classify a single requirement"""
        model, tokenizer = self.load_classifier()
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        # Map class to category (adjust based on model's actual classes)
        categories = ['Functional', 'Non-Functional', 'Performance', 'Security', 'Usability']
        category = categories[predicted_class] if predicted_class < len(categories) else 'Unknown'
        
        return {
            'category': category,
            'confidence': round(confidence, 3),
            'all_scores': {cat: round(prob, 3) for cat, prob in zip(categories, probs[0].tolist())}
        }
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        model, tokenizer = self.load_embedder()
        
        # Tokenize all texts
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]
        return float(similarity)
    
    def find_dependencies(self, requirements: List[Dict], threshold: float = 0.6) -> List[Dict]:
        """Find dependencies between requirements using embeddings"""
        texts = [req['text'] for req in requirements]
        embeddings = self.get_embeddings(texts)
        
        dependencies = []
        
        for i in range(len(requirements)):
            for j in range(i + 1, len(requirements)):
                similarity = self.calculate_similarity(embeddings[i], embeddings[j])
                
                if similarity >= threshold:
                    dep_type = 'strong' if similarity > 0.8 else 'moderate' if similarity > 0.7 else 'weak'
                    
                    dependencies.append({
                        'source_id': requirements[i]['req_id'],
                        'target_id': requirements[j]['req_id'],
                        'similarity_score': round(similarity, 3),
                        'dependency_type': dep_type
                    })
        
        return dependencies
    
    def analyze_financial_impact(self, text: str) -> Dict:
        """Analyze financial/business impact of a requirement"""
        model, tokenizer = self.load_financial_analyzer()
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        # Map to impact levels
        impact_levels = ['Low', 'Medium', 'High', 'Critical']
        impact = impact_levels[predicted_class] if predicted_class < len(impact_levels) else 'Unknown'
        
        return {
            'impact_level': impact,
            'confidence': round(confidence, 3),
            'risk_score': round(confidence * (predicted_class + 1) / len(impact_levels), 3)
        }
    
    def summarize_requirements(self, texts: List[str], max_length: int = 150) -> str:
        """Summarize multiple requirements"""
        summarizer = self.load_summarizer()
        
        # Combine texts
        combined = " ".join(texts)
        
        # Truncate if too long
        if len(combined) > 1000:
            combined = combined[:1000]
        
        try:
            summary = summarizer(
                combined,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def analyze_impact_chain(self, changed_req_id: str, requirements: List[Dict], 
                           dependencies: List[Dict], max_depth: int = 3) -> List[Dict]:
        """Analyze cascading impact of a requirement change"""
        impacted = []
        visited = set()
        
        def find_impacted(req_id: str, depth: int):
            if depth > max_depth or req_id in visited:
                return
            
            visited.add(req_id)
            
            # Find all dependencies involving this requirement
            for dep in dependencies:
                target_id = None
                similarity = dep['similarity_score']
                
                if dep['source_id'] == req_id:
                    target_id = dep['target_id']
                elif dep['target_id'] == req_id:
                    target_id = dep['source_id']
                
                if target_id and target_id not in visited:
                    # Calculate impact score based on similarity and depth
                    impact_score = similarity * (1 - depth * 0.2)  # Decay with depth
                    
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
                    
                    # Recursively find downstream impacts
                    find_impacted(target_id, depth + 1)
        
        find_impacted(changed_req_id, 1)
        
        # Sort by impact score
        impacted.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return impacted

# Create singleton instance
model_manager = ModelManager()