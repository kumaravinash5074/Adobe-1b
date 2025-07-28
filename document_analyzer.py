"""
Round 1B: Persona-Driven Document Intelligence
Theme: "Connect What Matters — For the User Who Matters"

This module extends the Round 1A PDF heading extractor to provide intelligent
document analysis based on persona and job-to-be-done requirements.
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import time
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import our Round 1A extractor
from app import PDFHeadingExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonaDocumentAnalyzer:
    """Intelligent document analyzer for persona-driven content extraction."""
    
    def __init__(self):
        self.input_dir = Path("/app/input")
        self.output_dir = Path("/app/output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.start_time = None
        
        # Initialize Round 1A extractor
        self.pdf_extractor = PDFHeadingExtractor()
        
        # Persona-specific keywords and patterns
        self.persona_keywords = {
            "researcher": ["methodology", "dataset", "benchmark", "performance", "evaluation", "results", "conclusion"],
            "student": ["concept", "mechanism", "reaction", "kinetics", "study", "learn", "understand", "exam"],
            "analyst": ["revenue", "trend", "investment", "market", "strategy", "financial", "growth", "competition"],
            "journalist": ["news", "event", "impact", "development", "announcement", "trend", "analysis"],
            "entrepreneur": ["business", "market", "opportunity", "strategy", "growth", "investment", "competition"],
            "travel planner": ["trip", "travel", "itinerary", "accommodation", "restaurant", "hotel", "attraction", "activity", "culture", "cuisine", "history", "tips", "tricks", "planning", "group", "friends", "days", "visit", "explore", "experience"]
        }
        
        # Initialize sentence transformer model for semantic similarity
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def start_timer(self):
        """Start performance timer."""
        self.start_time = time.time()
        
    def check_timeout(self, max_seconds=60):
        """Check if processing time exceeds limit."""
        if self.start_time and (time.time() - self.start_time) > max_seconds:
            raise TimeoutError(f"Processing exceeded {max_seconds} second limit")
    
    def extract_text_with_context(self, page, bbox=None) -> str:
        """Extract text from a page or specific region with context."""
        try:
            if bbox:
                # Extract from specific region
                text = page.get_text("text", clip=bbox)
            else:
                # Extract entire page
                text = page.get_text("text")
            return text.strip()
        except Exception as e:
            logger.warning(f"Error extracting text: {e}")
            return ""
    
    def calculate_section_relevance(self, section_text: str, persona: str, job_description: str) -> float:
        """Calculate relevance score for a section based on persona and job, using both keyword and semantic similarity."""
        relevance_score = 0.0
        # Get persona-specific keywords
        persona_keywords = self.persona_keywords.get(persona.lower(), [])
        # Extract keywords from job description
        job_keywords = self.extract_keywords_from_job(job_description)
        # Combine all relevant keywords
        all_keywords = persona_keywords + job_keywords
        # Calculate keyword matches
        section_lower = section_text.lower()
        keyword_matches = sum(1 for keyword in all_keywords if keyword.lower() in section_lower)
        # Base score from keyword matches
        relevance_score += keyword_matches * 0.2  # Lowered weight to balance with semantic
        # Boost for exact phrase matches
        for keyword in all_keywords:
            if keyword.lower() in section_lower:
                relevance_score += 0.1
        # Boost for section length (not too short, not too long)
        text_length = len(section_text)
        if 50 <= text_length <= 500:
            relevance_score += 0.05
        elif text_length > 500:
            relevance_score += 0.02
        # Boost for structured content (numbered lists, bullet points)
        if re.search(r'\d+\.|•|\*', section_text):
            relevance_score += 0.05
        # Boost for technical terms (capitalized words, abbreviations)
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', section_text))
        relevance_score += min(technical_terms * 0.02, 0.1)
        # --- Semantic similarity boost ---
        try:
            persona_job_query = persona + " " + job_description
            section_emb = self.semantic_model.encode([section_text])
            query_emb = self.semantic_model.encode([persona_job_query])
            semantic_sim = cosine_similarity(section_emb, query_emb)[0][0]  # 0-1
            # Weight semantic similarity as 60% of the final score
            relevance_score = 0.6 * semantic_sim + 0.4 * min(relevance_score, 1.0)
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            relevance_score = min(relevance_score, 1.0)
        return min(relevance_score, 1.0)  # Cap at 1.0
    
    def extract_keywords_from_job(self, job_description: str) -> List[str]:
        """Extract relevant keywords from job description."""
        # Common job-related keywords
        job_keywords = []
        
        # Extract nouns and important phrases
        words = re.findall(r'\b[A-Za-z]{3,}\b', job_description.lower())
        
        # Filter for meaningful words
        meaningful_words = [word for word in words if len(word) > 3 and word not in 
                          ['the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'will', 'should']]
        
        # Add specific job-related terms
        if 'literature' in job_description.lower():
            job_keywords.extend(['literature', 'review', 'research', 'paper', 'methodology'])
        if 'exam' in job_description.lower():
            job_keywords.extend(['exam', 'study', 'concept', 'mechanism', 'reaction'])
        if 'financial' in job_description.lower() or 'revenue' in job_description.lower():
            job_keywords.extend(['financial', 'revenue', 'trend', 'investment', 'market'])
        if 'trip' in job_description.lower() or 'travel' in job_description.lower():
            job_keywords.extend(['trip', 'travel', 'planning', 'itinerary', 'accommodation', 'restaurant', 'hotel', 'attraction', 'activity', 'group', 'friends', 'days', 'visit'])
        
        return list(set(meaningful_words + job_keywords))
    
    def extract_subsections(self, section_text: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract and analyze subsections from a section."""
        subsections = []
        
        # Split by common subsection patterns
        lines = section_text.split('\n')
        current_subsection = ""
        current_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a subsection header
            if (re.match(r'^\d+\.\d+', line) or  # 2.1, 2.2, etc.
                re.match(r'^[A-Z][A-Z\s]+$', line) or  # ALL CAPS
                re.match(r'^[A-Z][a-z\s]+$', line) or  # Title Case
                len(line) < 100 and line.endswith(':')):
                
                # Save previous subsection
                if current_text.strip():
                    subsections.append({
                        "document": "current",  # Will be updated by caller
                        "refined_text": current_text.strip(),
                        "page_number": page_num
                    })
                
                current_subsection = line
                current_text = line + "\n"
            else:
                current_text += line + "\n"
        
        # Add the last subsection
        if current_text.strip():
            subsections.append({
                "document": "current",  # Will be updated by caller
                "refined_text": current_text.strip(),
                "page_number": page_num
            })
        
        return subsections
    
    def analyze_documents(self, persona: str, job_description: str) -> Dict[str, Any]:
        """Main analysis method for persona-driven document intelligence."""
        logger.info(f"Starting analysis for persona: {persona}")
        logger.info(f"Job description: {job_description}")
        
        # Get all PDF files
        pdf_files = list(self.input_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError("No PDF files found in input directory")
        
        logger.info(f"Found {len(pdf_files)} PDF files to analyze")
        
        all_sections = []
        all_subsections = []
        
        # Process each PDF
        for pdf_path in pdf_files:
            self.check_timeout()
            
            logger.info(f"Analyzing document: {pdf_path.name}")
            
            try:
                doc = fitz.open(str(pdf_path))
                
                # Get document outline using Round 1A extractor
                outline_data = self.pdf_extractor.process_pdf(pdf_path)
                
                # Extract sections with context
                for heading in outline_data.get("outline", []):
                    self.check_timeout()
                    
                    page_num = heading["page"]
                    section_title = heading["text"]
                    
                    # Extract section content
                    page = doc.load_page(page_num - 1)  # 0-based indexing
                    section_text = self.extract_text_with_context(page)
                    
                    # Calculate relevance
                    relevance_score = self.calculate_section_relevance(section_text, persona, job_description)
                    
                    # Only include relevant sections (score > 0.1)
                    if relevance_score > 0.1:
                        section_info = {
                            "document": pdf_path.name,
                            "page_number": page_num,
                            "section_title": section_title,
                            "importance_rank": relevance_score
                        }
                        all_sections.append(section_info)
                        
                        # Extract subsections
                        subsections = self.extract_subsections(section_text, page_num)
                        for subsection in subsections:
                            subsection["document"] = pdf_path.name
                            all_subsections.append(subsection)
                
                doc.close()
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                continue
        
        # Sort sections by importance rank (descending)
        all_sections.sort(key=lambda x: x["importance_rank"], reverse=True)
        
        # Sort subsections by document and page
        all_subsections.sort(key=lambda x: (x["document"], x["page_number"]))
        
        # Prepare output
        output = {
            "metadata": {
                "input_documents": [pdf.name for pdf in pdf_files],
                "persona": persona,
                "job_to_be_done": job_description,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": all_sections[:20],  # Top 20 most relevant sections
            "sub_section_analysis": all_subsections[:50]  # Top 50 subsections
        }
        
        return output
    
    def save_output(self, data: Dict[str, Any], filename: str = "analysis_result.json"):
        """Save analysis results to JSON file."""
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved analysis results to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving output: {e}")
    
    def run(self):
        """Main execution method for Round 1B."""
        logger.info("Starting Persona-Driven Document Intelligence Analysis")
        self.start_timer()
        
        # Read persona and job description from input files
        persona_file = self.input_dir / "persona.txt"
        job_file = self.input_dir / "job.txt"
        
        if not persona_file.exists() or not job_file.exists():
            logger.error("Missing persona.txt or job.txt files in input directory")
            return
        
        try:
            with open(persona_file, 'r', encoding='utf-8') as f:
                persona = f.read().strip()
            
            with open(job_file, 'r', encoding='utf-8') as f:
                job_description = f.read().strip()
            
            logger.info(f"Persona: {persona}")
            logger.info(f"Job: {job_description}")
            
            # Perform analysis
            results = self.analyze_documents(persona, job_description)
            
            # Save results
            self.save_output(results)
            
            processing_time = time.time() - self.start_time if self.start_time else 0
            logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Save error output
            error_output = {
                "metadata": {
                    "input_documents": [],
                    "persona": persona if 'persona' in locals() else "Unknown",
                    "job_to_be_done": job_description if 'job_description' in locals() else "Unknown",
                    "processing_timestamp": datetime.now().isoformat(),
                    "error": str(e)
                },
                "extracted_sections": [],
                "sub_section_analysis": []
            }
            self.save_output(error_output, "error_result.json")

def main():
    """Main entry point for Round 1B."""
    analyzer = PersonaDocumentAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main() 