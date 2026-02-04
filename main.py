"""
Author: Tamasa Patra
Purpose: Command-line interface with multi-manual support
What it does:
    - Setup command now processes multiple PDFs from registry
    - Validates manual availability
    - Shows manual inventory
    - All other commands remain the same
"""

import os
import sys
from src.document_processor import DocumentProcessor
from src.chroma_client import ChromaDBManager
from src.langgraph_workflow import CoffeeMakerRAG
from src.manual_manager import ManualManager
from config.settings import settings  # Your existing config
from config.manual_registry import manual_registry
from langchain_openai import ChatOpenAI


# ========================================
# NEW: Multi-Manual Setup Commands
# ========================================

def setup_index():
    """Setup: Process multiple PDFs and create ChromaDB index."""
    print("=" * 70)
    print("MULTI-MANUAL INDEX CREATION")
    print("=" * 70)
    print()
    
    # Validate available manuals
    print("🔍 Checking available manuals...\n")
    validation = manual_registry.validate_manuals()
    
    print(f"📚 Total registered manuals: {validation['total_registered']}")
    print(f"✅ Available: {validation['available']}")
    print(f"❌ Missing: {validation['missing']}")
    print()
    
    # Show available manuals
    if validation['available_manuals']:
        print("📗 Available manuals:")
        for manual in validation['available_manuals']:
            print(f"   • {manual['title']}")
            print(f"     Type: {manual['equipment_type']}")
            print(f"     Path: {manual['path']}")
        print()
    
    # Show missing manuals
    if validation['missing_manuals']:
        print("⚠️  Missing manuals (will be skipped):")
        for manual in validation['missing_manuals']:
            print(f"   • {manual['title']}")
            print(f"     Expected at: {manual['path']}")
        print()
    
    # Confirm with user
    if validation['available'] == 0:
        print("❌ No manuals available to index!")
        print("Please add PDF files to the manuals/ directory.")
        return
    
    proceed = input(f"Proceed with indexing {validation['available']} manual(s)? [Y/n]: ").strip()
    if proceed.lower() == 'n':
        print("Cancelled.")
        return
    
    # Process available manuals
    print()
    manager = ManualManager()
    available_manuals = manual_registry.get_available_manuals()
    
    summary = manager.process_multiple_manuals(
        manuals=available_manuals,
        reset_collection=True
    )
    
    print("\n✅ Setup complete!")


def show_manual_inventory():
    """Show all registered manuals and their status."""
    print("=" * 70)
    print("MANUAL INVENTORY")
    print("=" * 70)
    print()
    
    validation = manual_registry.validate_manuals()
    
    print(f"📊 Total Registered: {validation['total_registered']}")
    print(f"✅ Available: {validation['available']}")
    print(f"❌ Missing: {validation['missing']}")
    print()
    
    # Group by equipment type
    all_manuals = manual_registry.get_all_manuals()
    by_type = {}
    for manual in all_manuals:
        if manual.equipment_type not in by_type:
            by_type[manual.equipment_type] = []
        by_type[manual.equipment_type].append(manual)
    
    # Display by type
    for equipment_type, manuals in by_type.items():
        print(f"\n{'─' * 70}")
        print(f"📁 {equipment_type}")
        print(f"{'─' * 70}")
        
        for manual in manuals:
            status = "✅" if manual.exists() else "❌"
            print(f"{status} {manual.title}")
            print(f"   Brand: {manual.equipment_brand} | Model: {manual.equipment_model}")
            print(f"   Type: {manual.manual_type} | Tier: {manual.tier}")
            print(f"   Path: {manual.pdf_path}")
            print()


def setup_manual_directories():
    """Create directory structure for manuals."""
    directories = [
        "manuals/coffee",
        "manuals/pos",
        "manuals/kitchen",
        "manuals/beverage"
    ]
    
    print("=" * 70)
    print("SETUP MANUAL DIRECTORIES")
    print("=" * 70)
    print()
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    # Create README
    readme_content = """# Equipment Manuals

## Directory Structure

- `coffee/` - Coffee makers, espresso machines
- `pos/` - POS systems, payment processing
- `kitchen/` - Ovens, fryers, grills, refrigerators
- `beverage/` - Ice machines, soda dispensers, blenders

## Adding New Manuals

1. Place PDF in appropriate directory
2. Update `config/manual_registry.py` with manual definition
3. Run `python main.py setup` to index

## Current Manuals

Run `python main.py show-manuals` to see inventory.
"""
    
    with open("manuals/README.md", "w") as f:
        f.write(readme_content)
    
    print("\n✅ Created manuals/README.md")
    print("\n📁 Directory structure ready!")
    print("   Add your PDF files and update config/manual_registry.py")


# ========================================
# EXISTING COMMANDS (Unchanged)
# ========================================

def test_queries():
    """Test the RAG system with sample queries."""
    print("=" * 70)
    print("TESTING RAG SYSTEM")
    print("=" * 70)
    print()
    
    chroma_manager = ChromaDBManager()
    chroma_manager.create_collection()
    stats = chroma_manager.get_collection_stats()
    
    if stats['count'] == 0:
        print("❌ No documents in ChromaDB!")
        print("Please run: python main.py setup")
        return
    
    print(f"✅ Found {stats['count']} documents in ChromaDB\n")
    
    rag = CoffeeMakerRAG()
    
    # Test questions - mix of equipment types
    questions = [
        # Coffee Maker questions
        "How do I descale the coffee maker?",
        "What should I do if the descale indicator light stays on?",
        
        # POS questions (if Oracle MICROS manual is available)
        "How do I close out the POS at end of day?",
        "How to process a refund?",
        
        # Multi-equipment question
        "What's the procedure for closing the restaurant?"
    ]
    
    for i, question in enumerate(questions, 1):
        print("\n" + "=" * 70)
        print(f"QUESTION {i}: {question}")
        print("=" * 70)
        print()
        
        result = rag.query(question)
        
        print(f"\n{'─' * 70}")
        print("📊 RESULTS")
        print(f"{'─' * 70}")
        print(f"Retries: {result['retries']}")
        print(f"Sources: {len(result['documents'])} documents")
        
        # Show equipment distribution
        if result['documents']:
            equipment_types = set([
                doc.get('equipment_type', 'unknown') 
                for doc in result['documents']
            ])
            print(f"Equipment types: {', '.join(equipment_types)}")
        
        print()
        print("💡 ANSWER:")
        print(f"{'─' * 70}")
        print(result['answer'])
        print(f"{'─' * 70}")
        
        if i < len(questions):
            input("\n⏎ Press Enter for next question...")


def interactive_mode():
    """Interactive Q&A mode."""
    print("=" * 70)
    print("INTERACTIVE MODE - Equipment Manual Assistant")
    print("=" * 70)
    print()
    print("Ask questions about your restaurant equipment")
    print("Type 'quit', 'exit', or 'q' to exit")
    print("Type 'stats' to see collection statistics")
    print()
    
    chroma_manager = ChromaDBManager()
    chroma_manager.create_collection()
    stats = chroma_manager.get_collection_stats()
    
    if stats['count'] == 0:
        print("❌ No documents in ChromaDB!")
        print("Please run: python main.py setup")
        return
    
    print(f"✅ Loaded {stats['count']} document chunks")
    print(f"✅ Model: {settings.openai_model}")
    print()
    
    rag = CoffeeMakerRAG()
    
    while True:
        print("─" * 70)
        question = input("\n🤔 Your question: ").strip()
        print()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if question.lower() == 'stats':
            stats = chroma_manager.get_collection_stats()
            print(f"📊 Collection Stats:")
            print(f"   Name: {stats['name']}")
            print(f"   Documents: {stats['count']}")
            print(f"   Location: {stats['persist_directory']}")
            continue
        
        if not question:
            continue
        
        print("⚙️  Processing...\n")
        result = rag.query(question)
        
        print("💡 ANSWER:")
        print("─" * 70)
        print(result['answer'])
        print("─" * 70)


# ========================================
# DEBUG COMMANDS (Keep existing)
# ========================================

def debug_chunks():
    """Show content of specific chunks."""
    print("=" * 70)
    print("DEBUG: INSPECT CHUNKS")
    print("=" * 70)
    print()
    
    chroma_manager = ChromaDBManager()
    chroma_manager.create_collection()
    
    stats = chroma_manager.get_collection_stats()
    if stats['count'] == 0:
        print("❌ No documents in ChromaDB!")
        print("Please run: python main.py setup")
        return
    
    print(f"📦 Total chunks in database: {stats['count']}\n")
    
    chunk_input = input("Enter chunk IDs to inspect (e.g., '53,54' or '28-32' or 'all'): ").strip()
    
    def clean_int(val):
        return int(val.strip().replace("'", "").replace('"', ""))
    
    chunk_ids = []
    
    try:
        if chunk_input.lower() == 'all':
            chunk_ids = list(range(stats['count']))
        elif '-' in chunk_input:
            start, end = chunk_input.split('-')
            chunk_ids = list(range(clean_int(start), clean_int(end) + 1))
        elif ',' in chunk_input:
            chunk_ids = [clean_int(x) for x in chunk_input.split(',')]
        else:
            chunk_ids = [clean_int(chunk_input)]
        
        print()
        
        for chunk_id in chunk_ids:
            try:
                results = chroma_manager.collection.get(
                    ids=[f"chunk_{chunk_id}"],
                    include=["documents", "metadatas"]
                )
                
                if results and results['documents']:
                    print("─" * 70)
                    print(f"CHUNK {chunk_id}")
                    print("─" * 70)
                    
                    content = results['documents'][0]
                    metadata = results['metadatas'][0] if results['metadatas'] else {}
                    
                    print(f"Equipment: {metadata.get('equipment_type', 'unknown')}")
                    print(f"Source: {metadata.get('source', 'unknown')}")
                    print(f"Page: {metadata.get('page_number', 'N/A')}")
                    print(f"Length: {len(content)} characters")
                    print()
                    print("Content:")
                    print(content)
                    print()
                else:
                    print(f"❌ Chunk {chunk_id} not found\n")
                    
            except Exception as e:
                print(f"❌ Error fetching chunk {chunk_id}: {e}\n")
                
    except ValueError as e:
        print(f"\n❌ Invalid Input Error: {e}")
        print("Please enter plain numbers (e.g., 50-55) without extra quotes.")
    
    print("=" * 70)


def search_for_keyword():
    """Find which chunks contain a specific keyword."""
    print("=" * 70)
    print("DEBUG: SEARCH FOR KEYWORD")
    print("=" * 70)
    print()
    
    chroma_manager = ChromaDBManager()
    chroma_manager.create_collection()
    
    stats = chroma_manager.get_collection_stats()
    if stats['count'] == 0:
        print("❌ No documents in ChromaDB!")
        print("Please run: python main.py setup")
        return
    
    keyword = input("Enter keyword to search for: ").strip()
    
    if not keyword:
        print("❌ No keyword entered")
        return
    
    print(f"\n🔍 Searching for chunks containing '{keyword}'...\n")
    
    all_docs = chroma_manager.collection.get(
        include=["documents", "metadatas"]
    )
    
    found_count = 0
    
    for i, doc in enumerate(all_docs['documents']):
        if keyword.lower() in doc.lower():
            found_count += 1
            metadata = all_docs['metadatas'][i]
            chunk_id = metadata.get('chunk_id', '?')
            equipment_type = metadata.get('equipment_type', 'unknown')
            
            lines = doc.split('\n')
            relevant_lines = [line for line in lines if keyword.lower() in line.lower()]
            
            print("─" * 70)
            print(f"CHUNK {chunk_id} ({equipment_type})")
            print("─" * 70)
            
            if relevant_lines:
                for line in relevant_lines[:3]:
                    print(f"   ...{line.strip()}...")
            else:
                preview = doc[:150].replace('\n', ' ')
                print(f"   {preview}...")
            
            print()
    
    print("=" * 70)
    print(f"✅ Found '{keyword}' in {found_count} chunks")
    print("=" * 70)


def test_retrieval():
    """Test how well retrieval works for different queries."""
    print("=" * 70)
    print("DEBUG: TEST RETRIEVAL QUALITY")
    print("=" * 70)
    print()
    
    chroma_manager = ChromaDBManager()
    chroma_manager.create_collection()
    
    stats = chroma_manager.get_collection_stats()
    if stats['count'] == 0:
        print("❌ No documents in ChromaDB!")
        print("Please run: python main.py setup")
        return
    
    custom_query = input("Enter test query (or press Enter for default tests): ").strip()
    
    if custom_query:
        queries = [custom_query]
    else:
        queries = [
            "descaling procedure",
            "how to close out POS",
            "process refund",
            "temperature settings",
            "end of day procedures"
        ]
    
    print()
    
    for query in queries:
        print("─" * 70)
        print(f"Query: '{query}'")
        print("─" * 70)
        
        docs = chroma_manager.search(query, top_k=5)
        
        if not docs:
            print("   ❌ No results found\n")
            continue
        
        print(f"   Retrieved {len(docs)} chunks:\n")
        
        for i, doc in enumerate(docs, 1):
            chunk_id = doc.get('chunk_id', '?')
            score = doc.get('score', 0)
            equipment_type = doc.get('equipment_type', 'unknown')
            
            print(f"   {i}. Chunk {chunk_id} ({equipment_type}) - similarity: {score:.3f}")
            
            #preview = doc['content'][:200].replace('\n', ' ')
            #print(f"      Preview: {preview}...")
            #print()
            meta = doc.get("metadata",{})
            title = meta.get("title", "Unknown Manual")
            page = meta.get("page_number","N/A")
            chunk_id = doc.get("chunk_id","?")
            score = doc.get("score",0)

            print(f"   {i}. Chunk {chunk_id} | {title} | Page {page} | similarity: {score:.3f}")
            preview = doc.get("content", "")[:800].replace("\n", " ")
            print(f"      Preview: {preview}...\n")








    
    print("=" * 70)


def simple_query(question: str = None):
    """Bypass grading, just retrieve and generate directly."""
    print("=" * 70)
    print("DEBUG: SIMPLE QUERY (NO GRADING)")
    print("=" * 70)
    print()
    
    chroma_manager = ChromaDBManager()
    chroma_manager.create_collection()
    
    stats = chroma_manager.get_collection_stats()
    if stats['count'] == 0:
        print("❌ No documents in ChromaDB!")
        print("Please run: python main.py setup")
        return
    
    if not question:
        question = input("Enter your question: ").strip()
    
    if not question:
        print("❌ No question entered")
        return
    
    print(f"\n🤔 Question: {question}\n")
    
    llm = ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        model=settings.openai_model
    )
    
    print("🔍 Retrieving documents...")
    docs = chroma_manager.search(question, top_k=5)
    
    print(f"✅ Retrieved {len(docs)} chunks:\n")
    for i, doc in enumerate(docs, 1):
        equipment_type = doc.get('equipment_type', 'unknown')
        print(f"   {i}. Chunk {doc['chunk_id']} ({equipment_type}): {doc['score']:.3f}")
    
    print("\n💭 Generating answer (bypassing grading step)...\n")
    
    context = "\n\n".join([
        f"[{doc.get('equipment_type', 'unknown')} - Source {i+1}]\n{doc['content']}"
        for i, doc in enumerate(docs)
    ])
    
    prompt = f"""Using this manual content, answer the question clearly and concisely.

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    
    print("─" * 70)
    print("💡 ANSWER:")
    print("─" * 70)
    print(response.content)
    print("─" * 70)
    print()


def show_help():
    """Show usage instructions."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║      Equipment Manual Assistant - Multi-Manual Edition               ║
║                                                                      ║
║  Stack: LangGraph + ChromaDB + OpenAI                               ║
╚══════════════════════════════════════════════════════════════════════╝

SETUP COMMANDS:

  python main.py setup-dirs        - Create manual directory structure
  python main.py show-manuals      - Show manual inventory
  python main.py setup             - Index all available manuals

USAGE COMMANDS:

  python main.py test              - Run test queries
  python main.py interactive       - Interactive Q&A mode

DEBUG COMMANDS:

  python main.py debug-chunks      - Inspect specific chunk contents
  python main.py search-keyword    - Find chunks containing a keyword
  python main.py test-retrieval    - Test retrieval quality
  python main.py simple-query      - Query without grading (debug)

HELP:

  python main.py help              - Show this help

GETTING STARTED:

1. python main.py setup-dirs       (creates folder structure)
2. Add your PDF files to manuals/ folders
3. Update config/manual_registry.py
4. python main.py show-manuals     (verify PDFs found)
5. python main.py setup            (index all manuals)
6. python main.py interactive      (start asking questions!)
    """)


# ========================================
# MAIN ENTRY POINT
# ========================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        # Setup commands
        if command == "setup-dirs":
            setup_manual_directories()
        elif command == "show-manuals":
            show_manual_inventory()
        elif command == "setup":
            setup_index()
        
        # Usage commands
        elif command == "test":
            test_queries()
        elif command == "interactive":
            interactive_mode()
        
        # Debug commands
        elif command == "debug-chunks":
            debug_chunks()
        elif command == "search-keyword":
            search_for_keyword()
        elif command == "test-retrieval":
            test_retrieval()
        elif command == "simple-query":
            question = sys.argv[2] if len(sys.argv) > 2 else None
            simple_query(question)
        
        # Help
        elif command == "help":
            show_help()
        
        else:
            print(f"❌ Unknown command: {command}")
            print("\nRun: python main.py help")
    else:
        show_help()
