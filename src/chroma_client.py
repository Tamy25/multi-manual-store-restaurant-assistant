"""
Author: Tamasa Patra
Purpose: Vector database operations (ChromaDB wrapper)
What it does:
Initializes ChromaDB persistent client
Generates embeddings using OpenAI API
Indexes document chunks as vectors
Performs similarity search
Manages collection (create/delete/stats)
Dependencies: chromadb, langchain-openai
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from config.settings import settings
import os
import uuid

class ChromaDBManager:
    """Manage ChromaDB vector store operations."""
    
    def __init__(self):
        print(f"🔧 Initializing ChromaDB...")
        
        # Create persist directory if it doesn't exist
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )
        
        # Get or create collection
        self.collection_name = settings.chroma_collection_name
        self.collection = None
        
        print(f"✅ ChromaDB initialized at {settings.chroma_persist_directory}")
    
    def create_collection(self, reset: bool = False):
        """Create or get collection."""
        if reset:
            print(f"🗑️  Deleting existing collection '{self.collection_name}'...")
            try:
                self.client.delete_collection(self.collection_name)
                print("✅ Collection deleted")
            except:
                pass
        
        print(f"📦 Creating collection '{self.collection_name}'...")
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"✅ Collection ready")
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata for ChromaDB.
        1. Remove keys with None values (ChromaDB crashes on None).
        2. Ensure all values are simple types (str, int, float, bool).
        """
        clean_metadata = {}
        for key, value in metadata.items():
            if value is not None:
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    # Convert unknown types (like lists or objects) to string
                    clean_metadata[key] = str(value)
        return clean_metadata

    def index_documents(self, documents: List[Dict]):
        """Index documents with embeddings."""
        if self.collection is None:
            self.create_collection()
        
        print(f"\n📊 Indexing {len(documents)} documents...")
        
        # Prepare data containers
        ids = []
        contents = []
        metadatas = []

        # Process each document
        for doc in documents:
            # FIX: Force a unique UUID for every chunk to prevent DuplicateIDError
            # The readable 'chunk_id' (e.g., 1, 2) is still saved in metadata
            unique_id = str(uuid.uuid4())
            ids.append(unique_id)
            
            contents.append(doc.get("content", ""))
            
            # Construct metadata dictionary
            raw_metadata = {
                "chunk_id": doc.get("chunk_id", 0),
                "source": doc.get("source", "unknown"),
                "title": doc.get("title", ""),
                "page_number": doc.get("page_number"),
                "manual_url": doc.get("manual_url"),
                "equipment_type": doc.get("equipment_type", "unknown"),
                "equipment_brand": doc.get("equipment_brand", "unknown"),
                "equipment_model": doc.get("equipment_model", "unknown")
            }
            
            # SANITIZE metadata to remove None values
            metadatas.append(self._sanitize_metadata(raw_metadata))
        
        # Generate embeddings
        print("🔮 Generating embeddings...")
        embeddings = self.embeddings.embed_documents(contents)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=contents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            print(f"  Indexed {end_idx}/{len(documents)} documents...")
        
        print(f"✅ All documents indexed successfully!")
    
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        equipment_brand: Optional[str] = None,
        equipment_type: Optional[str] = None,
    ) -> List[Dict]:
        """Search with optional metadata filtering (brand/type) + safe fallbacks."""
        if self.collection is None:
            self.create_collection()
    
        def _run_query(where_filter: Optional[Dict[str, Any]], n: int) -> List[Dict]:
            if where_filter:
                print(f"🔍 Filtering by: {where_filter}")
    
            query_embedding = self.embeddings.embed_query(query)
    
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": n,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                search_kwargs["where"] = where_filter
    
            results = self.collection.query(**search_kwargs)
    
            documents = []
            if results and results.get("ids") and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    meta = results["metadatas"][0][i] or {}
                    documents.append({
                        "content": results["documents"][0][i],
                        "metadata": meta,
                        "score": 1 - results["distances"][0][i] if "distances" in results else 0,
                        "chunk_id": meta.get("chunk_id", 0),
                        "source": meta.get("source", "unknown"),
                        "page_number": meta.get("page_number", "N/A"),
                        "equipment_type": meta.get("equipment_type", "unknown"),
                        "equipment_brand": meta.get("equipment_brand", "unknown"),
                        "title": meta.get("title", "Unknown Manual"),
                    })
            return documents
    
        # --------- Build where filters ----------
        def _where(brand: Optional[str], etype: Optional[str]) -> Optional[Dict[str, Any]]:
            conditions = []
            if brand:
                conditions.append({"equipment_brand": brand})
            if etype:
                conditions.append({"equipment_type": etype})
            if len(conditions) == 1:
                return conditions[0]
            if len(conditions) > 1:
                return {"$and": conditions}
            return None
    
        # 1) strict: brand + type
        docs = _run_query(_where(equipment_brand, equipment_type), top_k)
    
        # 2) fallback: type only (very important for your case!)
        if not docs and equipment_type:
            print("⚠️ 0 docs with brand+type. Retrying with type-only...")
            docs = _run_query(_where(None, equipment_type), top_k)
    
        # 3) fallback: no filter
        if not docs:
            print("⚠️ 0 docs with filters. Retrying with NO filters...")
            docs = _run_query(None, top_k)
    
        return docs
    

    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        if self.collection is None:
             self.create_collection()
        
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "persist_directory": settings.chroma_persist_directory
        }
    
    def delete_collection(self):
        """Delete the collection."""
        if self.collection:
            self.client.delete_collection(self.collection_name)
            self.collection = None
            print(f"✅ Collection '{self.collection_name}' deleted")


    def debug_metadata(self, where: Optional[Dict[str, Any]] = None, limit: int = 20):
        """Print a quick metadata sample + distinct brand/type values."""
        if self.collection is None:
            self.create_collection()
    
        print("\n================= CHROMA METADATA DEBUG =================")
        print("Collection:", self.collection_name)
        if where:
            print("WHERE:", where)
    
        try:
            res = self.collection.get(where=where, include=["metadatas"], limit=limit)
        except TypeError:
            # Some Chroma versions don't support limit in get()
            res = self.collection.get(where=where, include=["metadatas"])
    
        metas = res.get("metadatas") or []
        metas = metas[:limit]
    
        print(f"Fetched {len(metas)} metadata records (showing up to {limit})")
    
        brands = set()
        types = set()
    
        for i, m in enumerate(metas[:10], 1):
            m = m or {}
            b = m.get("equipment_brand")
            t = m.get("equipment_type")
            brands.add(b)
            types.add(t)
            print(f"{i}. equipment_brand={b!r}, equipment_type={t!r}, title={m.get('title')!r}, page={m.get('page_number')!r}")
    
        print("\nDistinct brands (sample):", brands)
        print("Distinct types  (sample):", types)
        print("=========================================================\n")
    
