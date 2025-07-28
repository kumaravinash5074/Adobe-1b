# Round 1B: Persona-Driven Document Intelligence
## Approach Explanation

### Overview
This solution implements intelligent document analysis that extracts and prioritizes the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done. The system combines traditional keyword matching with advanced semantic similarity to provide highly relevant and contextually appropriate results.

### Methodology

#### 1. **Hybrid Relevance Scoring**
The system uses a **two-tier approach** for calculating section relevance:

**Keyword-Based Scoring (40% weight):**
- Persona-specific keywords (e.g., "trip", "travel", "itinerary" for travel planners)
- Job description keyword extraction using NLP patterns
- Exact phrase matching and technical term detection
- Structural content analysis (numbered lists, bullet points)

**Semantic Similarity (60% weight):**
- Uses `sentence-transformers` (all-MiniLM-L6-v2 model, ~80MB)
- Encodes both section text and persona+job query into high-dimensional vectors
- Computes cosine similarity for semantic relevance
- Handles synonyms, related concepts, and contextual meaning

#### 2. **Robust Heading Detection**
- **Scoring-based approach** combining font size, boldness, ALL CAPS, numbered patterns
- **Position-aware detection** using vertical spacing and alignment
- **Noise filtering** for dates, version numbers, and body text
- **Fragmented title handling** by combining text spans based on font size and proximity

#### 3. **Subsection Analysis**
- **Pattern-based extraction** using regex for numbered subsections, bullet points
- **Content refinement** to remove noise and improve readability
- **Hierarchical organization** maintaining document structure

#### 4. **Performance Optimizations**
- **Early timeout detection** (60 seconds max for 3-5 documents)
- **Efficient text extraction** using PyMuPDF's optimized methods
- **Smart filtering** to reduce processing overhead
- **Memory-efficient semantic model** (under 200MB total)

### Technical Implementation

#### Dependencies
- **PyMuPDF==1.23.8**: High-performance PDF processing
- **sentence-transformers==2.2.2**: Semantic similarity computation
- **scikit-learn==1.3.2**: Cosine similarity and vector operations

#### Architecture
- **CPU-only processing** (no GPU required)
- **Offline operation** (no internet access during execution)
- **AMD64 compatibility** for cross-platform deployment
- **Docker containerization** for consistent execution environment

### Key Innovations

1. **Semantic-Aware Ranking**: Goes beyond keyword matching to understand context and meaning
2. **Persona-Specific Optimization**: Tailored keyword sets and relevance patterns for different user types
3. **Robust Title Extraction**: Handles fragmented and multi-line titles common in real-world PDFs
4. **Scalable Processing**: Efficiently handles 3-10 documents within time constraints

### Output Quality
- **Top 20 most relevant sections** ranked by combined relevance score
- **Top 50 subsections** with refined text and page numbers
- **Comprehensive metadata** including processing timestamp and input validation
- **Error handling** with graceful degradation and informative error messages

This approach ensures that the system can generalize across diverse document types, personas, and job requirements while maintaining high accuracy and performance standards. 