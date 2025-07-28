# Adobe Hackathon Round 1B: Persona-Driven Document Intelligence

## Overview
This solution analyzes a collection of PDFs and extracts the most relevant sections based on a specific persona and job-to-be-done. It uses both keyword and semantic similarity for high-quality, context-aware results.

## Features
- **Hybrid relevance scoring**: Combines persona/job keywords (40%) and semantic similarity (60%).
- **Robust heading detection**: Same multi-heuristic approach as Round 1A.
- **Fragmented title handling**: Aggregates multi-line or split title spans.
- **Noise filtering**: Ignores irrelevant content.
- **Comprehensive output**: Includes metadata, ranked sections, and sub-section analysis.
- **Performance**: ≤ 60 seconds for 3-5 documents, CPU-only, no internet required.

## Directory Structure
```
adobe-round1b/
├── document_analyzer.py
├── app.py
├── requirements.txt
├── Dockerfile
├── approach_explanation.md
├── README.md
├── input/      # Place your test PDFs, persona.txt, and job.txt here
└── output/     # Output JSON will be generated here
```

## Approach

### Hybrid Relevance Scoring
- **Keyword-based**: Persona-specific and job-extracted keywords, phrase matching, technical terms, structure.
- **Semantic similarity**: Uses `sentence-transformers` (all-MiniLM-L6-v2) for deep context matching.

### Heading & Title Extraction
- Same robust, scoring-based approach as Round 1A.
- Handles fragmented and multi-line titles.

### Subsection Analysis
- Pattern-based extraction (numbered, bulleted, etc.)
- Refined text and hierarchical organization.

## Input Requirements
The `/app/input` directory **must contain**:
1. **3-10 PDF documents** to analyze
2. **persona.txt** — Description of the user persona (e.g., `PhD Researcher in Computational Biology`)
3. **job.txt** — Specific task description (e.g., `Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks`)

> **Note:** The system will log an error and exit if either `persona.txt` or `job.txt` is missing.

### Sample persona.txt
```
PhD Researcher in Computational Biology
```
### Sample job.txt
```
Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks
```

## How to Build and Run

### Build the Docker Image
```bash
docker build --platform linux/amd64 -t round1b-persona-analyzer .
```

### Run the Container
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none round1b-persona-analyzer
```
- On Windows PowerShell, use `${PWD}` instead of `$(pwd)`.

### Expected Execution
- Reads persona and job description from `/app/input/persona.txt` and `/app/input/job.txt`
- Processes all PDF documents in `/app/input`
- Extracts and ranks sections based on persona/job relevance
- Generates `analysis_result.json` in `/app/output`

## Output Format
```json
{
  "metadata": {
    "input_documents": ["file01.pdf", "file02.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks",
    "processing_timestamp": "2025-07-27T14:12:01.776"
  },
  "extracted_sections": [
    {
      "document": "file01.pdf",
      "page_number": 1,
      "section_title": "Introduction",
      "importance_rank": 0.85
    }
  ],
  "sub_section_analysis": [
    {
      "document": "file01.pdf",
      "refined_text": "Section content...",
      "page_number": 1
    }
  ]
}
```

## Constraints & Compliance
- **Execution time**: ≤ 60 seconds for 3-5 documents
- **Model size**: ≤ 1GB (including sentence-transformers)
- **Architecture**: CPU-only, AMD64 compatible
- **Network**: No internet access required
- **No hardcoding**: No file-specific logic

## Troubleshooting & FAQ
- **Missing persona/job file?** The system will log an error and exit.
- **No output?** Check that your PDFs, persona.txt, and job.txt are in the `input/` directory.
- **Output not as expected?** Review your persona/job description for clarity and specificity.

## Contact
For any issues, please contact the author or open an issue in your private repository. 