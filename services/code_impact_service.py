"""
Code Impact Service - Intelligent codebase analysis
Predicts which files/functions will be affected by a requirement change
"""
from typing import List, Dict, Any
from models.model_manager import model_manager
import re
import os

class CodeImpactService:
    """Analyzes codebase and predicts impact of changes"""
    
    def __init__(self):
        self.model_manager = model_manager
        
    def analyze_codebase_impact(self, requirement: str, codebase_files: List[Dict]) -> Dict:
        """
        Main function: Analyze which files will be affected by a requirement
        
        Args:
            requirement: Natural language requirement (e.g., "add fingerprint authentication")
            codebase_files: List of {name: str, content: str, type: str}
        
        Returns:
            {
                impact: "Positive/Negative/Neutral",
                affected_files: [...],
                suggested_changes: [...],
                reasoning: "..."
            }
        """
        try:
            print(f"🔍 Analyzing impact of: {requirement}")
            
            # Step 1: Classify requirement
            classification = self.model_manager.classify_requirement(requirement)
            
            # Step 2: Analyze sentiment/impact
            impact_result = self.model_manager.analyze_impact(requirement)
            
            # ✅ FIX 3: Override impact if needed (fix neutral → positive for security)
            original_impact = impact_result.get('impact', 'Neutral')
            corrected_impact = self._override_impact_classification(requirement, original_impact)
            
            # Step 3: Build codebase context
            codebase_context = self._build_codebase_context(codebase_files)
            
            # Step 4: Use StarCoder2 to predict affected files
            affected_analysis = self._predict_affected_files(
                requirement, 
                codebase_context, 
                codebase_files
            )
            
            # Step 5: Generate specific suggestions
            suggestions = self._generate_suggestions(
                requirement,
                affected_analysis['affected_files'],
                codebase_files
            )
            
            return {
                "requirement": requirement,
                "classification": classification.get('classification', 'Unknown'),
                "classification_confidence": classification.get('confidence', 0),
                "impact": corrected_impact,  # ✅ Use corrected impact
                "impact_confidence": min(impact_result.get('confidence', 0), 85),  # ✅ FIX 4: Cap at 85%
                "affected_files": affected_analysis['affected_files'],
                "suggested_changes": suggestions,
                "reasoning": affected_analysis['reasoning'],
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Code impact analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def _override_impact_classification(self, requirement: str, original_impact: str) -> str:
        """
        ✅ NEW METHOD - Override impact classification for known positive patterns
        Fixes issue where security improvements are marked as 'neutral'
        """
        req_lower = requirement.lower()
        
        # Positive indicators - security, performance, user experience improvements
        positive_keywords = [
            'security', 'authentication', 'fingerprint', 'biometric',
            'encryption', 'protection', 'secure', 'safety',
            'improve', 'enhance', 'optimize', 'upgrade', 'better',
            'performance', 'faster', 'efficiency',
            'user experience', 'usability', 'accessibility'
        ]
        
        # Check if requirement contains positive keywords
        if any(keyword in req_lower for keyword in positive_keywords):
            # But avoid false positives for removal/deprecation
            negative_indicators = ['remove', 'delete', 'deprecate', 'disable', 'break']
            if not any(neg in req_lower for neg in negative_indicators):
                return "Positive"
        
        # Keep original classification if no override needed
        return original_impact
    
    def _build_codebase_context(self, files: List[Dict]) -> str:
        """Build a structured context of the codebase"""
        context = "=== CODEBASE STRUCTURE ===\n\n"
        
        for file in files:
            name = file.get('name', 'unknown')
            content = file.get('content', '')
            
            # Extract key information
            functions = self._extract_functions(content, name)
            classes = self._extract_classes(content, name)
            
            context += f"File: {name}\n"
            if classes:
                context += f"  Classes: {', '.join(classes)}\n"
            if functions:
                context += f"  Functions: {', '.join(functions)}\n"
            context += "\n"
        
        return context
    
    def _extract_functions(self, code: str, filename: str) -> List[str]:
        """Extract function names from code"""
        functions = []
        
        if filename.endswith('.py'):
            functions = re.findall(r'def\s+(\w+)\s*\(', code)
        elif filename.endswith(('.js', '.ts', '.jsx', '.tsx')):
            functions = re.findall(r'function\s+(\w+)\s*\(', code)
            functions += re.findall(r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', code)
        elif filename.endswith('.java'):
            functions = re.findall(r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(', code)
        
        return functions[:10]
    
    def _extract_classes(self, code: str, filename: str) -> List[str]:
        """Extract class names from code"""
        classes = []
        
        if filename.endswith(('.py', '.java', '.js', '.ts')):
            classes = re.findall(r'class\s+(\w+)', code)
        
        return classes
    
    def _predict_affected_files(self, requirement: str, codebase_context: str, files: List[Dict]) -> Dict:
        """Use StarCoder2 to predict which files will be affected"""
        
        # Create a focused prompt for StarCoder2
        prompt = f"""Given this requirement change:
"{requirement}"

And this codebase structure:
{codebase_context}

Task: Identify which files will be affected and why.

Format your response as:
AFFECTED FILES:
- filename1.py: reason why it's affected
- filename2.py: reason why it's affected

REASONING:
Brief explanation of the overall impact
"""
        
        # Get StarCoder2 analysis
        code_analysis = self.model_manager.analyze_code(prompt)
        
        # Parse the response to extract affected files
        affected_files = self._parse_affected_files(code_analysis, files, requirement)
        
        # Extract reasoning
        reasoning = self._extract_reasoning(code_analysis, requirement)
        
        return {
            "affected_files": affected_files,
            "reasoning": reasoning
        }
    
    def _parse_affected_files(self, analysis: str, files: List[Dict], requirement: str) -> List[Dict]:
        """✅ FIX 1: Parse StarCoder2 output to extract affected files (no duplicates)"""
        affected = []
        affected_names = set()  # ✅ Track which files we've already added
        available_filenames = [f['name'] for f in files]
        
        # Look for file patterns in the analysis
        for filename in available_filenames:
            # ✅ Skip if already added
            if filename in affected_names:
                continue
                
            base_name = filename.replace('.py', '').replace('.js', '').replace('.java', '')
            
            # Check if file is mentioned in analysis
            if filename.lower() in analysis.lower() or base_name.lower() in analysis.lower():
                reason = "Mentioned in impact analysis"
                
                # Look for specific patterns
                if 'auth' in filename.lower():
                    reason = "Authentication logic needs modification"
                elif 'user' in filename.lower():
                    reason = "User model requires updates"
                elif 'database' in filename.lower() or 'schema' in filename.lower():
                    reason = "Database schema changes required"
                
                affected.append({
                    "file": filename,
                    "reason": reason,
                    "confidence": 0.85
                })
                affected_names.add(filename)  # ✅ Mark as added
        
        # If no files detected, make educated guesses
        if not affected:
            affected = self._guess_affected_files(requirement, files)
        
        return affected
    
    def _guess_affected_files(self, requirement: str, files: List[Dict]) -> List[Dict]:
        """Fallback: Guess affected files based on keywords"""
        affected = []
        req_lower = requirement.lower()
        
        # Keyword mapping
        keywords = {
            'auth': ['auth', 'login', 'session'],
            'user': ['user', 'profile', 'account'],
            'database': ['db', 'schema', 'sql', 'model'],
            'api': ['api', 'endpoint', 'route', 'controller'],
        }
        
        for file in files:
            filename_lower = file['name'].lower()
            
            for category, patterns in keywords.items():
                if category in req_lower:
                    for pattern in patterns:
                        if pattern in filename_lower:
                            affected.append({
                                "file": file['name'],
                                "reason": f"Related to {category} functionality",
                                "confidence": 0.7
                            })
                            break
        
        return affected[:5]
    
    def _extract_reasoning(self, analysis: str, requirement: str) -> str:
        """Extract reasoning from StarCoder2 output"""
        if "REASONING:" in analysis:
            reasoning = analysis.split("REASONING:")[1].strip()
            return reasoning[:300]
        
        return f"Based on the requirement '{requirement}', the AI has identified files that likely need modification to implement this change."
    
    def _generate_suggestions(self, requirement: str, affected_files: List[Dict], files: List[Dict]) -> List[str]:
        """✅ FIX 2: Generate specific code change suggestions (no duplicates)"""
        suggestions = []
        suggestions_set = set()  # ✅ Track unique suggestions
        req_lower = requirement.lower()
        
        for affected in affected_files:
            filename = affected['file']
            
            if 'auth' in filename.lower():
                if 'fingerprint' in req_lower:
                    sugg1 = f"In {filename}: Replace email authentication with fingerprint-based authentication"
                    sugg2 = f"In {filename}: Add fingerprint_hash verification method"
                    if sugg1 not in suggestions_set:
                        suggestions.append(sugg1)
                        suggestions_set.add(sugg1)
                    if sugg2 not in suggestions_set:
                        suggestions.append(sugg2)
                        suggestions_set.add(sugg2)
                else:
                    sugg = f"In {filename}: Update authentication flow"
                    if sugg not in suggestions_set:
                        suggestions.append(sugg)
                        suggestions_set.add(sugg)
            
            elif 'user' in filename.lower() or 'model' in filename.lower():
                if 'fingerprint' in req_lower:
                    sugg = f"In {filename}: Add fingerprint_hash field to User model"
                    if sugg not in suggestions_set:
                        suggestions.append(sugg)
                        suggestions_set.add(sugg)
                else:
                    sugg = f"In {filename}: Update data model schema"
                    if sugg not in suggestions_set:
                        suggestions.append(sugg)
                        suggestions_set.add(sugg)
            
            elif 'database' in filename.lower() or 'schema' in filename.lower():
                if 'fingerprint' in req_lower:
                    sugg1 = f"In {filename}: Add fingerprint_hash column to users table"
                    sugg2 = f"In {filename}: Create index on fingerprint_hash for faster lookups"
                    if sugg1 not in suggestions_set:
                        suggestions.append(sugg1)
                        suggestions_set.add(sugg1)
                    if sugg2 not in suggestions_set:
                        suggestions.append(sugg2)
                        suggestions_set.add(sugg2)
                else:
                    sugg = f"In {filename}: Update database schema"
                    if sugg not in suggestions_set:
                        suggestions.append(sugg)
                        suggestions_set.add(sugg)
        
        if not suggestions:
            suggestions.append("Analyze the requirement and implement necessary changes")
            suggestions.append("Update tests to cover new functionality")
        
        return suggestions[:5]

# Create singleton instance
code_impact_service = CodeImpactService()