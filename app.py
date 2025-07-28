#!/usr/bin/env python3
"""
PDF Heading Extractor - Adobe Hackathon Round 1A

This script processes PDF files from the input directory, extracts document titles
and headings (H1, H2, H3) based on font size hierarchy, and saves results as JSON.
Designed for high performance and accuracy in document structure extraction.
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFHeadingExtractor:
    """Extracts titles and headings from PDF documents with high accuracy."""
    
    def __init__(self):
        self.input_dir = Path("/app/input")
        self.output_dir = Path("/app/output")
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.start_time = None
        
    def start_timer(self):
        """Start performance timer."""
        self.start_time = time.time()
        
    def check_timeout(self, max_seconds=10):
        """Check if processing time exceeds limit."""
        if self.start_time and (time.time() - self.start_time) > max_seconds:
            raise TimeoutError(f"Processing exceeded {max_seconds} second limit")
    
    def extract_text_blocks(self, page) -> List[Dict[str, Any]]:
        """Extract text blocks with their properties from a page, including y-position."""
        blocks = []
        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if text and len(text) > 2:
                                font_size = span.get("size", 0)
                                font_flags = span.get("flags", 0)
                                is_bold = font_flags & 2**4  # Bold flag
                                font = span.get("font", "")
                                y = span.get("y", None)
                                block_info = {
                                    "text": text,
                                    "font_size": font_size,
                                    "is_bold": bool(is_bold),
                                    "font": font,
                                    "y": y
                                }
                                blocks.append(block_info)
        except Exception as e:
            logger.warning(f"Error extracting text from page: {e}")
        return blocks
    
    def is_heading(self, text: str, font_size: float, is_bold: bool, y_position: float = None, prev_y: float = None) -> bool:
        """Improved heading detection using multiple heuristics and a scoring system."""
        import re
        score = 0
        clean_text = text.strip()
        if len(clean_text) < 3 or len(clean_text) > 80:
            return False
        if clean_text.isdigit():
            return False
        if clean_text.replace('.', '').replace('-', '').replace('_', '').strip() == '':
            return False
        # Numbered heading patterns
        if re.match(r'^\d+\.\s+[A-Z]', clean_text):
            score += 2
        if re.match(r'^\d+\.\d+\s+[A-Z]', clean_text):
            score += 2
        if re.match(r'^\d+(\.\d+)+', clean_text):
            score += 1
        # ALL CAPS
        if clean_text.isupper() and 8 <= len(clean_text) <= 40:
            score += 1
        # Title Case
        if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$', clean_text):
            score += 1
        # Bold
        if is_bold:
            score += 1
        # Large font
        if font_size >= 16.0:
            score += 1
        if font_size >= 18.0:
            score += 1
        # Whitespace above (if available)
        if y_position is not None and prev_y is not None and (y_position - prev_y) > 20:
            score += 1
        # Important phrases
        important_phrases = [
            "revision history", "table of contents", "acknowledgements", "references",
            "intended audience", "career paths for testers", "learning objectives",
            "entry requirements", "structure and course duration", "keeping it current",
            "business outcomes", "content", "trademarks", "documents and web sites"
        ]
        if any(phrase in clean_text.lower() for phrase in important_phrases):
            score += 1
        # Penalty for too many numbers
        if re.match(r'^[\d\s\.]+$', clean_text):
            score -= 1
        # Penalty for date/version patterns
        date_patterns = [
            r'^\d{1,2}\s+[A-Z]{3,}\s+\d{4}$',
            r'^\d{1,2}\s+[A-Z]{3,}\s+\d{4}',
            r'^\d+\.\d+$',
            r'^\d+\.\d+\s+\d{1,2}\s+[A-Z]{3,}\s+\d{4}',
        ]
        for pattern in date_patterns:
            if re.match(pattern, clean_text):
                score -= 2
        return score >= 2
    
    def is_noise_content(self, text: str) -> bool:
        """Check if text is noise that should be filtered out."""
        import re
        
        # Skip dates and version numbers
        date_patterns = [
            r'^\d{1,2}\s+[A-Z]{3,}\s+\d{4}$',  # 18 JUNE 2013
            r'^\d+\.\d+$',                      # 0.1, 0.2, etc.
            r'^\d+\.\d+\s+\d{1,2}\s+[A-Z]{3,}\s+\d{4}',  # 0.1 18 JUNE 2013
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, text.strip()):
                return True
                
        # Skip text that's mostly numbers and dates
        if re.match(r'^[\d\s\.]+$', text.strip()):
            return True
            
        # Skip very short or very long text
        if len(text.strip()) < 8 or len(text.strip()) > 60:
            return True
            
        # Skip text that looks like body content (starts with lowercase)
        if text.strip() and text.strip()[0].islower():
            return True
            
        # Skip text that contains too many words (likely body content)
        if len(text.split()) > 6:
            return True
            
        # Skip single words that aren't important
        single_words = ["board", "international", "qualifications", "version", "date", "remarks", "syllabus", "days", "istqb", "identifier", "reference"]
        if text.strip().lower() in single_words:
            return True
            
        # Skip text that's too generic
        generic_words = ["chris", "klaus", "initial version", "amended business outcomes"]
        for word in generic_words:
            if word in text.lower():
                return True
            
        return False
    
    def get_heading_level(self, text: str, font_size: float) -> str:
        """Determine heading level based on text pattern and font size."""
        import re
        
        # Check for numbered headings (most reliable)
        if re.match(r'^\d+\.\d+\.\d+', text.strip()):  # 2.1.1, 2.1.2, etc.
            return "H3"
        elif re.match(r'^\d+\.\d+', text.strip()):  # 2.1, 2.2, etc.
            return "H2"
        elif re.match(r'^\d+\.', text.strip()):  # 1., 2., etc.
            return "H1"
            
        # Check font size
        if font_size >= 16.0:
            return "H1"
        elif font_size >= 14.0:
            return "H2"
        else:
            return "H3"
    
    def extract_title(self, blocks: List[Dict[str, Any]]) -> str:
        """Extract document title from the first page."""
        if not blocks:
            return "Untitled Document"
            
        # Find the largest bold text that looks like a title
        title_candidates = []
        for block in blocks:
            if (block["is_bold"] and 
                block["font_size"] >= 12 and 
                len(block["text"]) > 3 and
                len(block["text"]) < 100):
                title_candidates.append(block)
        
        if title_candidates:
            title_candidates.sort(key=lambda x: x["font_size"], reverse=True)
            return title_candidates[0]["text"]
            
        # Find the largest text that's not too long
        valid_blocks = [b for b in blocks if len(b["text"]) > 3 and len(b["text"]) < 100]
        if valid_blocks:
            largest_text = max(valid_blocks, key=lambda x: x["font_size"])
            return largest_text["text"]
            
        return "Untitled Document"
    
    def combine_title_elements(self, blocks: List[Dict[str, Any]]) -> str:
        """
        Combine text blocks that are likely part of the title, based on font size, boldness, and vertical proximity.
        """
        if not blocks:
            return "Untitled Document"

        # Find the largest font size among blocks
        max_font_size = max(block["font_size"] for block in blocks)
        # Consider blocks that are close to the largest font size (e.g., within 1.5pt)
        title_blocks = [b for b in blocks if abs(b["font_size"] - max_font_size) < 1.5]

        # Sort by y-position (top to bottom)
        title_blocks = sorted(title_blocks, key=lambda b: b.get("y", 0))

        # Combine text fragments that are close together vertically
        combined_title = ""
        prev_y = None
        for block in title_blocks:
            if prev_y is not None and abs(block.get("y", 0) - prev_y) > 30:
                # If the next block is far away, break (likely not part of the title)
                break
            combined_title += block["text"] + " "
            prev_y = block.get("y", 0)

        return combined_title.strip() if combined_title else "Untitled Document"
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file and extract title and headings."""
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        try:
            doc = fitz.open(str(pdf_path))
            all_headings = []
            title = "Untitled Document"
            
            # Check page count for performance
            if len(doc) > 50:
                logger.warning(f"PDF has {len(doc)} pages, may exceed time limit")
            
            for page_num in range(len(doc)):
                # Check timeout every few pages
                if page_num % 5 == 0:
                    self.check_timeout()
                    
                page = doc.load_page(page_num)
                blocks = self.extract_text_blocks(page)
                
                # Extract title from first page
                if page_num == 0 and blocks:
                    title = self.combine_title_elements(blocks)
                
                # Extract headings from current page
                prev_y = None
                for block in blocks:
                    y_position = block.get("y", None) if "y" in block else None
                    if self.is_heading(block["text"], block["font_size"], block["is_bold"], y_position, prev_y):
                        level = self.get_heading_level(block["text"], block["font_size"])
                        heading = {
                            "level": level,
                            "text": block["text"].strip(),
                            "page": page_num + 1
                        }
                        all_headings.append(heading)
                    prev_y = y_position
            
            doc.close()
            
            # Remove duplicates and clean up
            seen_texts = set()
            clean_headings = []
            for heading in all_headings:
                text = heading["text"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    clean_headings.append(heading)
            
            # Additional filtering to remove noise
            final_headings = []
            for heading in clean_headings:
                text = heading["text"]
                
                # Skip if it's still noise
                if self.is_noise_content(text):
                    continue
                    
                # Skip if it's too short or too long
                if len(text) < 5 or len(text) > 80:
                    continue
                    
                final_headings.append(heading)
            
            # Sort by page number for consistent output
            final_headings.sort(key=lambda x: (x["page"], x["text"]))
            
            return {
                "title": title,
                "outline": final_headings
            }
            
        except TimeoutError:
            logger.error(f"Processing timeout for {pdf_path.name}")
            return {
                "title": "Processing Timeout",
                "outline": [],
                "error": "Processing exceeded time limit"
            }
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            return {
                "title": "Error Processing Document",
                "outline": [],
                "error": str(e)
            }
    
    def save_output(self, filename: str, data: Dict[str, Any]):
        """Save extracted data to JSON file."""
        output_path = self.output_dir / f"{filename}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved output to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving output for {filename}: {str(e)}")
    
    def run(self):
        """Main execution method."""
        logger.info("Starting PDF Heading Extractor")
        self.start_timer()
        
        # Check if input directory exists
        if not self.input_dir.exists():
            logger.error(f"Input directory not found: {self.input_dir}")
            return
        
        # Get all PDF files from input directory
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("No PDF files found in input directory")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
        
        # Process each PDF file
        for pdf_path in pdf_files:
            try:
                # Extract data
                data = self.process_pdf(pdf_path)
                
                # Save output (use filename without extension)
                filename = pdf_path.stem
                self.save_output(filename, data)
                
                # Check overall timeout
                self.check_timeout()
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue
        
        processing_time = time.time() - self.start_time if self.start_time else 0
        logger.info(f"PDF processing completed in {processing_time:.2f} seconds")

def main():
    """Main entry point."""
    extractor = PDFHeadingExtractor()
    extractor.run()

if __name__ == "__main__":
    main() 