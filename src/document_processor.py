"""
Author: Tamasa Patra
Purpose: PDF text extraction and document chunking
What it does:
Extracts text from PDF files (page by page)
Splits text into manageable chunks using RecursiveCharacterTextSplitter
Adds metadata (chunk_id, source, title, page_number) to each chunk
Key classes: DocumentProcessor
Key methods:
extract_text_from_pdf() - Extracts raw text + tracks page positions
chunk_document() - Splits into chunks with overlap + assigns page numbers
process_pdf() - Complete pipeline
_find_page_for_position() - Helper to map character position to page number
Dependencies: PyPDF2, langchain-text-splitters
Lines: ~120
"""
import PyPDF2
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

class DocumentProcessor:
    """Process and chunk PDF documents for indexing."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[int, Tuple[int, int]]]:
        """
        Extract text from PDF file and track page positions.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            tuple: (full_text, page_map)
                - full_text: Complete extracted text with page markers
                - page_map: Dict mapping {page_number: (start_pos, end_pos)}
        """
        print(f"📄 Extracting text from {pdf_path}...")
        text = ""
        page_map = {}  # Track character positions for each page
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    # Record start position before adding page content
                    start_pos = len(text)
                    
                    # Extract page text
                    page_text = page.extract_text()
                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                    
                    # Record end position after adding page content
                    end_pos = len(text)
                    
                    # Store page boundaries (1-indexed page numbers)
                    page_map[page_num + 1] = (start_pos, end_pos)
                    
                    if (page_num + 1) % 5 == 0:
                        print(f"  Processed {page_num + 1}/{total_pages} pages...")
                
                print(f"✅ Extracted {len(text)} characters from {total_pages} pages")
                print(f"✅ Tracked page positions for {len(page_map)} pages")
                
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            raise
        
        return text, page_map
    
    def _find_page_for_position(
        self, 
        char_position: int, 
        page_map: Dict[int, Tuple[int, int]]
    ) -> int:
        """
        Find which page a character position belongs to.
        
        Args:
            char_position: Character position in the full text
            page_map: Dict of {page_num: (start_pos, end_pos)}
        
        Returns:
            Page number (1-indexed), or 1 if not found
        """
        for page_num, (start_pos, end_pos) in page_map.items():
            if start_pos <= char_position < end_pos:
                return page_num
        
        # If position not found (edge case), return page 1
        return 1
    
    def chunk_document(
        self, 
        text: str, 
        page_map: Dict[int, Tuple[int, int]] = None,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Split document into chunks with metadata.
        
        Args:
            text: Full document text
            page_map: Optional dict mapping page numbers to char positions
            metadata: Optional metadata to add to each chunk
            
        Returns:
            List of document chunks with metadata
        """
        print(f"✂️  Splitting into chunks...")
        
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        current_position = 0  # Track position in original text for page mapping
        
        for i, chunk in enumerate(chunks):
            # Determine page number for this chunk
            page_number = None
            if page_map:
                # Find the page where this chunk starts
                page_number = self._find_page_for_position(current_position, page_map)
            
            # Create document with basic fields
            doc = {
                "id": f"chunk_{i}",
                "content": chunk,
                "chunk_id": i,
                "total_chunks": len(chunks)
            }
            
            # Add page number if available
            if page_number is not None:
                doc["page_number"] = page_number
            
            # Update position for next chunk
            # Note: Chunks may overlap, but we use start position for page assignment
            current_position += len(chunk)
            
            # Add any additional metadata passed in
            if metadata:
                doc.update(metadata)
            
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} chunks")
        if page_map:
            print(f"✅ Assigned page numbers to chunks")
        
        return documents
    
    def process_pdf(
        self, 
        pdf_path: str, 
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Complete pipeline: extract text, track pages, and chunk document.
        
        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata to add to each chunk
            
        Returns:
            List of document chunks with page numbers and metadata
        """
        # Extract text with page tracking
        text, page_map = self.extract_text_from_pdf(pdf_path)
        
        # Chunk with page number assignment
        documents = self.chunk_document(
            text=text,
            page_map=page_map,
            metadata=metadata
        )
        
        return documents
