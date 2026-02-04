"""
Author: Tamasa Patra
Purpose: Multi-manual processing coordinator
What it does:
    - Coordinates processing of multiple PDF manuals
    - Tracks processing progress
    - Handles errors gracefully for individual manuals
    - Provides batch indexing statistics
"""

from typing import List, Dict
from config.manual_registry import ManualDefinition
from src.document_processor import DocumentProcessor
from src.chroma_client import ChromaDBManager


class ManualManager:
    """Coordinate processing and indexing of multiple manuals."""
    
    def __init__(self):
        """Initialize manual manager."""
        self.processor = DocumentProcessor()
        self.chroma_manager = ChromaDBManager()
    
    def process_manual(
        self, 
        manual: ManualDefinition,
        verbose: bool = True
    ) -> Dict:
        """
        Process a single manual.
        
        Args:
            manual: ManualDefinition object
            verbose: Print progress messages
            
        Returns:
            Dict with processing results
        """
        result = {
            "manual": manual.title,
            "success": False,
            "chunks_created": 0,
            "error": None
        }
        
        try:
            if verbose:
                print(f"\n{'=' * 70}")
                print(f"📄 Processing: {manual.title}")
                print(f"   File: {manual.pdf_path}")
                print(f"   Type: {manual.equipment_type} ({manual.manual_type})")
                print(f"{'=' * 70}")
            
            # Check if file exists
            if not manual.exists():
                raise FileNotFoundError(f"PDF not found: {manual.pdf_path}")
            
            # Process PDF with metadata
            documents = self.processor.process_pdf(
                pdf_path=manual.pdf_path,
                metadata=manual.to_metadata()
            )
            
            result["chunks_created"] = len(documents)
            result["documents"] = documents
            result["success"] = True
            
            if verbose:
                print(f"✅ Successfully created {len(documents)} chunks")
            
        except Exception as e:
            result["error"] = str(e)
            if verbose:
                print(f"❌ Error processing manual: {e}")
        
        return result
    
    def process_multiple_manuals(
        self, 
        manuals: List[ManualDefinition],
        reset_collection: bool = True
    ) -> Dict:
        """
        Process multiple manuals and index them.
        
        Args:
            manuals: List of ManualDefinition objects
            reset_collection: Whether to reset ChromaDB collection first
            
        Returns:
            Dict with batch processing statistics
        """
        print("\n" + "=" * 70)
        print("MULTI-MANUAL PROCESSING")
        print("=" * 70)
        print(f"📚 Manuals to process: {len(manuals)}")
        print()
        
        # Reset collection if requested
        if reset_collection:
            print("🗑️  Resetting ChromaDB collection...")
            self.chroma_manager.create_collection(reset=True)
            print("✅ Collection reset\n")
        else:
            self.chroma_manager.create_collection()
        
        # Track results
        all_documents = []
        successful = []
        failed = []
        
        # Process each manual
        for i, manual in enumerate(manuals, 1):
            print(f"\n[{i}/{len(manuals)}] ", end="")
            
            result = self.process_manual(manual, verbose=True)
            
            if result["success"]:
                successful.append(manual.title)
                all_documents.extend(result["documents"])
            else:
                failed.append({
                    "title": manual.title,
                    "error": result["error"]
                })
        
        # Index all documents at once
        if all_documents:
            print(f"\n{'=' * 70}")
            print(f"📊 INDEXING ALL DOCUMENTS")
            print(f"{'=' * 70}")
            print(f"Total chunks to index: {len(all_documents)}\n")
            
            self.chroma_manager.index_documents(all_documents)
        
        # Generate summary
        stats = self.chroma_manager.get_collection_stats()
        
        summary = {
            "total_manuals": len(manuals),
            "successful": len(successful),
            "failed": len(failed),
            "total_chunks": len(all_documents),
            "indexed_chunks": stats["count"],
            "successful_manuals": successful,
            "failed_manuals": failed,
            "collection_stats": stats
        }
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print processing summary."""
        print("\n" + "=" * 70)
        print("📊 PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Total Manuals: {summary['total_manuals']}")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📦 Total Chunks: {summary['total_chunks']}")
        print(f"💾 Indexed: {summary['indexed_chunks']}")
        print()
        
        if summary['successful_manuals']:
            print("✅ Successfully processed:")
            for manual in summary['successful_manuals']:
                print(f"   • {manual}")
            print()
        
        if summary['failed_manuals']:
            print("❌ Failed to process:")
            for item in summary['failed_manuals']:
                print(f"   • {item['title']}")
                print(f"     Error: {item['error']}")
            print()
        
        print(f"📍 Collection: {summary['collection_stats']['name']}")
        print(f"📂 Location: {summary['collection_stats']['persist_directory']}")
        print("=" * 70)
