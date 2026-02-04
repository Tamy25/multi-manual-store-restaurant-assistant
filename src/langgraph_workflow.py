"""
Author: Tamasa Patra
Purpose: RAG orchestration and LangGraph workflow definition
What it does:
Defines workflow state (GraphState)
Implements retrieve → generate workflow
Handles query rewriting (optional)
Manages LLM-based answer generation
Builds and compiles LangGraph state machine
Key classes:
GraphState (TypedDict) - Workflow state definition
CoffeeMakerRAG - Main RAG system class
Key methods:
retrieve() - Document retrieval from ChromaDB
generate_answer() - LLM-based answer generation
rewrite_query() - Query optimization
_build_workflow() - LangGraph construction
query() - Main entry point
Dependencies: langgraph, langchain-openai, langchain-core
Lines: ~200+
Notes: Contains critical prompt engineering for answer generation"""

from typing import TypedDict, List, Annotated, Optional
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from numpy import rint
from src.chroma_client import ChromaDBManager
from config.settings import settings

# Define State
class GraphState(TypedDict):
    """State for the RAG workflow."""
    question: str
    generation: str
    documents: List[dict]
    retry_count: int
    #relevance_score: float
    rewritten_question: str
    short_ctx: dict  # {"last_question": str, "last_answer": str, "last_manual_title": str, ...}
    locked_brand: Optional[str]
    locked_type: Optional[str]
    locked_title: Optional[str]
    primary_manual_title: Optional[str]
    primary_equipment_type: Optional[str]
    primary_equipment_brand: Optional[str]

class CoffeeMakerRAG:
    """LangGraph-based RAG workflow for Coffee Maker Manual."""
    
    def __init__(self):
        print("🚀 Initializing Coffee Maker RAG system...")
        
        self.chroma_manager = ChromaDBManager()
        self.chroma_manager.create_collection()  # Ensure collection exists
        
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0
        )
        
        print(f"✅ Using model: {settings.openai_model}")
        print(f"✅ Using embeddings: {settings.openai_embedding_model}")
        
        self.workflow = self._build_workflow()
        print("✅ RAG system ready!\n")



    def retrieve(self, state: GraphState) -> GraphState:
        """
        Two-Stage Retrieval: Let the documents vote on equipment type.
        
        Stage 1: Search ALL manuals without filters
        Stage 2: Analyze results to detect primary equipment
        Stage 3: (Optional) Re-search with filter for cleaner results
        
        This eliminates hardcoded keyword lists - the semantic search naturally
        finds the right manual, and we just need to identify which one won.
        """
        print("---RETRIEVE DOCUMENTS---")
    
        question = state.get("rewritten_question") or state["question"]
        query_lower = question.lower()
    
        print(f"📝 RAW QUESTION: '{question}'")
        print(f"📝 QUESTION MARK COUNT: {question.count('?')}")
    
        # Detect multi-part questions and increase retrieval
        base_top_k = settings.retrieval_top_k  # Default: 8
        question_count = question.count('?')
        
        if question_count > 1:
            top_k = base_top_k * question_count
            print(f"📝 Multi-part question detected ({question_count} questions) → top_k increased to {top_k}")
        else:
            top_k = base_top_k
    
        # =========================================================================
        # Check for context lock (follow-up questions)
        # =========================================================================
        locked_brand = state.get("locked_brand")
        locked_type = state.get("locked_type")
        locked_title = state.get("locked_title")
    
        # If we have a conversation lock, use it directly
        if locked_brand or locked_type:
            print(f"🔒 Using locked context: brand={locked_brand}, type={locked_type}, title={locked_title}")
            
            documents = self.chroma_manager.search(
                query=question,
                top_k=top_k,
                equipment_brand=locked_brand,
                equipment_type=locked_type,
            )
            
            print(f"Retrieved {len(documents)} documents (from locked context)")
            
            primary = self._pick_primary_manual(documents)
            
            return {
                **state,
                "documents": documents,
                "locked_brand": locked_brand,
                "locked_type": locked_type,
                "locked_title": locked_title,
                "primary_manual_title": primary.get("title"),
                "primary_equipment_type": locked_type or primary.get("equipment_type"),
                "primary_equipment_brand": locked_brand or primary.get("equipment_brand"),
            }
    
        # =========================================================================
        # STAGE 1: Search ALL manuals (no filter) - let semantic search do the work
        # =========================================================================
        print("🔍 STAGE 1: Searching ALL manuals (no filter)...")

        
        # Search with more results for voting
        stage1_top_k = max(top_k * 2, 16)  # Get more for better voting
        
        all_results = self.chroma_manager.search(
            query=question,
            top_k=stage1_top_k,
            equipment_brand=None,
            equipment_type=None,
        )

        for i, doc in enumerate(all_results[:5]):
                    metadata = doc.get('metadata', {})
                    equip_type = metadata.get('equipment_type') or doc.get('equipment_type')
                    brand = metadata.get('equipment_brand') or doc.get('equipment_brand')
                    score = doc.get('score', 0)
                    print(f"   [{i+1}] {equip_type}/{brand} | Score: {score:.3f} | {doc.get('content', '')[:50]}...")
                
        
        print(f"   Retrieved {len(all_results)} documents from all manuals")
    
        # =========================================================================
        # STAGE 2: Analyze results - let documents vote on equipment type
        # =========================================================================
        print("📊 STAGE 2: Analyzing results to detect equipment type...")
        
        # Count equipment types in top results (weighted by rank)
        equipment_votes = {}
        brand_votes = {}
        
        for rank, doc in enumerate(all_results[:12]):  # Use top 12 for voting
            metadata = doc.get('metadata', {})
            equip_type = metadata.get('equipment_type') or doc.get('equipment_type')
            brand = metadata.get('equipment_brand') or doc.get('equipment_brand')
            
            # Weight by rank (higher rank = more weight)
            weight = 12 - rank  # First result gets weight 12, second gets 11, etc.
            
            if equip_type:
                equipment_votes[equip_type] = equipment_votes.get(equip_type, 0) + weight
            if brand:
                brand_votes[brand] = brand_votes.get(brand, 0) + weight
        
        print(f"   Equipment votes: {equipment_votes}")
        print(f"   Brand votes: {brand_votes}")
        
        # Determine winning equipment type and brand
        detected_type = None
        detected_brand = None
        
        if equipment_votes:
            detected_type = max(equipment_votes, key=equipment_votes.get)
            total_votes = sum(equipment_votes.values())
            confidence = equipment_votes[detected_type] / total_votes if total_votes > 0 else 0
            print(f"   🏆 Detected equipment type: {detected_type} (confidence: {confidence:.1%})")
        
        if brand_votes:
            detected_brand = max(brand_votes, key=brand_votes.get)
            print(f"   🏆 Detected brand: {detected_brand}")
    
        # =========================================================================
        # STAGE 3: Decide whether to re-filter or use mixed results
        # =========================================================================
        
        # Check if there's a clear winner (dominant equipment type)
        if equipment_votes:
            total_votes = sum(equipment_votes.values())
            winner_votes = equipment_votes.get(detected_type, 0)
            dominance = winner_votes / total_votes if total_votes > 0 else 0
            
            if dominance >= 0.6:  # 60%+ of top results are from one equipment type
                # Clear winner - re-search with filter for cleaner results
                print(f"🔍 STAGE 3: Clear winner ({dominance:.0%}) - filtering to {detected_type}...")
                
                documents = self.chroma_manager.search(
                    query=question,
                    top_k=top_k,
                    equipment_brand=detected_brand,
                    equipment_type=detected_type,
                )
                print(f"   Retrieved {len(documents)} filtered documents")
            else:
                # Mixed results - might be ambiguous query, use top results as-is
                print(f"🔍 STAGE 3: Mixed results ({dominance:.0%}) - using unfiltered top {top_k}")
                documents = all_results[:top_k]
        else:
            # No equipment detected - use unfiltered results
            print("🔍 STAGE 3: No equipment detected - using unfiltered results")
            documents = all_results[:top_k]
    
        print(f"📦 Final document count: {len(documents)}")
    
        # =========================================================================
        # Determine primary manual info for context tracking
        # =========================================================================
        primary = self._pick_primary_manual(documents)
        
        primary_type = detected_type or primary.get("equipment_type")
        primary_brand = detected_brand or primary.get("equipment_brand")
        primary_title = primary.get("title")
    
        # Log final distribution
        final_distribution = {}
        for doc in documents:
            metadata = doc.get('metadata', {})
            equip_type = metadata.get('equipment_type') or doc.get('equipment_type', 'unknown')
            final_distribution[equip_type] = final_distribution.get(equip_type, 0) + 1
        print(f"📊 Final distribution: {final_distribution}")
    
        return {
            **state,
            "documents": documents,
            
            # Set locks based on detected equipment (for follow-ups)
            "locked_brand": primary_brand,
            "locked_type": primary_type,
            "locked_title": primary_title,
            
            # Feed primary info forward + back to UI
            "primary_manual_title": primary_title,
            "primary_equipment_type": primary_type,
            "primary_equipment_brand": primary_brand,
        }
    
    
    # =============================================================================
    # HELPER METHOD: _pick_primary_manual (if you don't already have it)
    # =============================================================================
    
    def _pick_primary_manual(self, documents: list) -> dict:
        """
        Pick the primary manual from retrieved documents based on frequency.
        
        Returns dict with: title, equipment_type, equipment_brand
        """
        if not documents:
            return {"title": None, "equipment_type": None, "equipment_brand": None}
        
        # Count occurrences
        title_counts = {}
        type_counts = {}
        brand_counts = {}
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            
            title = metadata.get('title') or doc.get('title')
            equip_type = metadata.get('equipment_type') or doc.get('equipment_type')
            brand = metadata.get('equipment_brand') or doc.get('equipment_brand')
            
            if title:
                title_counts[title] = title_counts.get(title, 0) + 1
            if equip_type:
                type_counts[equip_type] = type_counts.get(equip_type, 0) + 1
            if brand:
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        return {
            "title": max(title_counts, key=title_counts.get) if title_counts else None,
            "equipment_type": max(type_counts, key=type_counts.get) if type_counts else None,
            "equipment_brand": max(brand_counts, key=brand_counts.get) if brand_counts else None,
        }

    
       
    def rewrite_query(self, state: GraphState) -> GraphState:
        """Rewrite query for better retrieval."""
        print(f"\n---REWRITE QUERY---")
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        
        rewrite_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a question rewriter for a coffee maker manual assistant.
            
            Rewrite the user's question to be more specific and likely to retrieve relevant information from a coffee maker manual.
            Include technical terms, model numbers if implied, and specific components.
            
            Return ONLY the rewritten question."""),
            HumanMessage(content=f"Original question: {question}\n\nRewritten question:")
        ])
        
        response = self.llm.invoke(rewrite_prompt.format_messages())
        rewritten_question = response.content.strip()
        
        print(f"Original: {question}")
        print(f"Rewritten: {rewritten_question}")
        
        return {
            **state,
            "rewritten_question": rewritten_question,
            "retry_count": retry_count + 1
        }
    

    def _generate_references(self, documents: List[dict]) -> str:
        """
    Generate reference section from retrieved documents with page numbers and links.
    Handles MULTIPLE manuals properly.
    
    Args:
        documents: List of retrieved document chunks with metadata
        
    Returns:
        Formatted reference string with page numbers from all manuals
    """
        if not documents:
            print("⚠️ No documents provided for references")
            return ""
    
        # Group pages by manual title
        manual_pages = {}  # {manual_title: set of page numbers}
    
        print(f"\n🔍 DEBUG: Generating references from {len(documents[:5])} documents:")
    
        for i, doc in enumerate(documents[:5]):  # Use top 5 sources
            print(f"\n  📄 Doc {i+1}:")
        
            # Extract metadata
            metadata = doc.get('metadata', {})
        
            # Get title and page
            title = metadata.get('title', 'Unknown Manual')
            page_num = metadata.get('page_number')
        
            print(f"     Title: {title}")
            print(f"     Page: {page_num}")
        
            if title and page_num is not None:
                if title not in manual_pages:
                    manual_pages[title] = set()
                manual_pages[title].add(page_num)
    
        # Build reference string
        if not manual_pages:
            print("⚠️ WARNING: No page numbers found in any documents!")
            return ""
    
        reference_parts = []
    
        for manual_title, pages in sorted(manual_pages.items()):
            sorted_pages = sorted(list(pages))
            print(f"\n✅ {manual_title}: {len(sorted_pages)} page(s) - {sorted_pages}")
        
            # Format page reference
            if len(sorted_pages) == 1:
                page_ref = f"Page {sorted_pages[0]}"
            elif len(sorted_pages) == 2:
                page_ref = f"Pages {sorted_pages[0]}, {sorted_pages[1]}"
            elif len(sorted_pages) <= 5:
                page_list = ', '.join(map(str, sorted_pages))
                page_ref = f"Pages {page_list}"
            else:
                # Check if consecutive
                is_consecutive = all(
                    sorted_pages[i] + 1 == sorted_pages[i + 1] 
                    for i in range(len(sorted_pages) - 1)
                )
            
                if is_consecutive:
                    page_ref = f"Pages {sorted_pages[0]}-{sorted_pages[-1]}"
                else:
                    page_list = ', '.join(map(str, sorted_pages[:3]))
                    page_ref = f"Pages {page_list}, and {len(sorted_pages) - 3} more"
        
            reference_parts.append(f"{manual_title}, {page_ref}")
    
        # Combine all references
        if len(reference_parts) == 1:
            reference = f"\n\n📖 **Reference:** {reference_parts[0]}"
        else:
            # Multiple manuals
            reference = "\n\n📖 **References:**\n"
            for ref in reference_parts:
                reference += f"   • {ref}\n"
    
        return reference



    def generate_answer(self, state: GraphState) -> GraphState:
        """Generate answer using retrieved documents."""
        print(f"\n---GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]
    
        # Use more documents for context
        max_docs = min(len(documents), 12)
        context = "\n\n".join([
            f"[Source {i+1}]\n{doc['content']}"
            for i, doc in enumerate(documents[:max_docs])
        ])
    
        # System prompt (stored as variable for proper length measurement)
        system_prompt = """You are an experienced Store Equipment Assistant 15+ years of hands-on experience supporting restaurants, 
        cafés, and retail stores. You have deep, practical knowledge of:
•        Daily store operations
•        Equipment usage and troubleshooting
•        Safety and compliance on the shop floor
•        Training busy store associates and operators store associates use and troubleshoot store equipment using ONLY the provided manual excerpts.
You help staff use and troubleshoot store equipment using ONLY the provided manual excerpts (the context).
    
You behave like a senior in-store expert, not a generic AI assistant.

PRIMARY GOAL
Provide clear, operator-friendly, step-by-step guidance that is:
•        Fully grounded in the provided manual excerpts
•        Practical for real store environments
•        Immediately usable by busy staff
❌ Do not give generic advice
❌ Do not rely on outside knowledge
✅ Do exactly what the manual supports

TONE & COMMUNICATION STYLE
Friendly but efficient (operators are busy)
Operational and practical.
Practical and action-oriented
Safety-conscious when relevant
Calm and safety-aware when relevant
No fluff
Confident, experienced
Speak like a senior colleague helping on the shop floor


    HARD CONSTRAINTS (NON-NEGOTIABLE):
    
    1) Manual-first grounding:
       - Use ONLY information explicitly present in the provided manual excerpts (the context).
       - Do NOT invent procedures, parts, error codes, locations, Menu paths, Button namesor troubleshooting steps that 
       are not in the excerpts. If it is not in the excerpts, you cannot assume it.
    
    2) UI/menu/Label exactness:
       - When instructions involve screens, buttons, or menus, copy the exact wording from the manual excerpt.
       - Preserve formatting and capitalization exactly.
       - Example labels to preserve exactly: "Configure network", "Settings > Network", "Wi-Fi toggle switch", "+ icon", "check mark".
       - Do NOT substitute with generic labels like "go to settings" unless the manual uses that exact wording.
       ❌ Do NOT paraphrase UI labels
       ❌ Do NOT use generic terms like “go to settings” unless the manual says that exactly
    
    3) NEVER say "check the manual" or use vague references:
       - ❌ FORBIDDEN PHRASES Never Use These): "as per the instructions", "as outlined in the manual",
         "refer to the manual", "follow the procedure", "complete the procedure as described", "see the manual", “As outlined in the manual”
       - You ARE the manual. Extract and provide the actual steps. Do not refer users elsewhere.
       - BAD: "Complete the descaling procedure as outlined in the manual."
       - GOOD: "To descale: (1) Dissolve 50g scale remover in 0.5L warm water (60-70°C)..."
       - Bad example: Complete the descaling procedure as outlined in the manual.”
       - Good example: “To descale: (1) Dissolve 50g scale remover in 0.5L warm water (60-70°C)...
         Step 1: Dissolve 50 g of scale remover in 0.5 L of warm water (60–70 °C).
         Step 2: Pour the solution into the descaling opening (8)."
    
    4) ALWAYS include specific values from the context:
       - Exact temperatures: "325°F (163°C)"
       - Exact times: "6 seconds", "15-30 minutes"
       - Exact quantities: "50g", "0.5L"
       - Exact button/part names: "programme button (5.4)", "descaling opening (8)"
    
    5) When explaining a procedure, include ALL prerequisite steps:
       - If resetting requires descaling first, explain the descaling steps too
       - Don't assume the user knows earlier steps
    
    6) If information is missing from context, say so clearly:
       - "I don't have that specific information in the retrieved sections."
       - Never tell users to "check the manual" - you ARE the manual

    7) # 11) When information is PARTIALLY available:
       - If the user asks about something and you find RELATED but not EXACT information, share what you found
       - Example: User asks "how to drain water" but manual only says "water remains in heating system"
       - BAD: "I don't have that specific information."
       - GOOD: "The manual doesn't include a draining procedure. It states that water always remains in the heating system and 
       warns not to place the appliance where temperatures fall below freezing. The overflow device discharges at the bottom of the 
       appliance. For safe draining before moving, I recommend contacting Metos support or a qualified technician."
    
    OUTPUT FORMAT (USE THESE EXACT SECTION HEADERS):
    
    1) Summary
       Start with a friendly acknowledgment like:
       #   - "Sure, I can help with that!"
       #   - "Got it!"
       #   - "Absolutely!"
       #   - "No problem!"
       # - Then briefly restate what the user needs in a warm, conversational tone
       # - Sound like a helpful colleague, not a robot
       # - Examples:
       #   ❌ BAD (robotic): "You want to print the transaction totals report."
       #   ✅ GOOD (friendly): "Sure, I can help you print the transaction totals report!"
       #   ✅ GOOD (friendly): "Got it! You're looking to print your transaction totals."
       #   ✅ GOOD (friendly): "Absolutely! Let me walk you through printing the totals report."
    
    2) Steps
       - Use numbered steps: Step 1, Step 2, Step 3...
       - Keep each step to 1-2 short sentences.
       - Use exact UI/menu/button labels from the excerpts.
       - Include at least 4-5 steps for each procedure when available in context.
       - Include at least 4–5 steps when the manual supports it
    
    3) Safety
       - Include ONLY when relevant (manual mentions hazards or the action implies risk).
       - Use ⚠️ and keep it to 1-3 bullets.
       - Only include hazards mentioned or implied by the manual
    
    4) Follow-ups
       - Ask 2-3 short, specific questions that help complete or confirm the task.

    5) When the user responds with just "yes", "no", "ok", "sure", "yeah", "yep", or similar:
- Do NOT guess what they mean
- Do NOT start a new unrelated topic (like descaling when they asked about draining)
- Instead, present the previous follow-up questions as numbered options:
  "Sure! Which would you like help with?
   1. [Follow-up question 1]
   2. [Follow-up question 2]  
   3. [Follow-up question 3]
   4. [Follow-up question 4]

   Just reply with 1, 2, or 3 or 4
   
   6) Follow-ups
    # - Ask 2-3 genuine follow-up QUESTIONS that help complete the task
    # - These should be actual questions, NOT numbered options
    # - Examples:
    #   ✅ GOOD: "Do you need help with maintenance tasks after draining?"
    #   ✅ GOOD: "Are you experiencing any leaks or issues?"
    #   ❌ BAD: "1. How to clean  2. How to descale  3. General tips"

#    ONLY convert to numbered options when user responds with "yes", "ok", "sure" etc.!
    
    STYLE RULES:
    - No long paragraphs. Use short lines and bullets.
    - Use emojis sparingly: ⚠️ for warnings, ✅ for confirmations, 💡 for tips
     Short lines and bullets only
    •        Emojis:
    o        ⚠️ for warnings
    o        ✅ for confirmations
    o        💡 for tips
    •        Avoid robotic phrasing
    •        Be calm, precise, and operational
    """
    
        # Check for multi-part question
        question_count = question.count('?')
        
        # Build user message - add explicit instruction for multi-part questions
        if question_count > 1:
            user_message = f"""Context:
    {context}
    
    Question: {question}
    
    ⚠️ CRITICAL - MULTI-PART QUESTION DETECTED ({question_count} parts):
    You MUST answer EACH part FULLY and SEPARATELY:
    
    1. Use clear section headers for each part (e.g., "**Lighting the Oven:**" and "**Baking Cookies:**")
    2. Provide COMPLETE steps for EACH question (minimum 4-5 steps per procedure)
    3. Do NOT abbreviate one answer to make room for another
    4. Include specific values (temperatures, times, button names) for EACH part
    5. Treat this as {question_count} separate questions that each deserve a full answer
    
    Now provide COMPLETE answers for ALL {question_count} parts:"""
        else:
            user_message = f"""Context:
    {context}
    
    Question: {question}
    
    Provide a complete answer with specific values and steps:"""
    
        # Build the prompt
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_message)
        ])
        
        # Debug prints
        print(f"📝 SYSTEM PROMPT LENGTH: {len(system_prompt)} chars")
        print(f"📝 CONTEXT LENGTH: {len(context)} chars")
        print(f"📝 QUESTION COUNT: {question_count}")
        print(f"📝 USER MESSAGE LENGTH: {len(user_message)} chars")
        
        # Call LLM
        response = self.llm.invoke(
            answer_prompt.format_messages(),
            max_tokens=3000
        )
    
        answer = response.content.strip()
        print(f"Generated answer length: {len(answer)} chars")
    
        # Generate and append reference section
        references = self._generate_references(documents)
    
        if references:
            answer = answer + references
            print(f"✅ Added references with page numbers")
    
        return {**state, "generation": answer}
    def decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to generate answer or rewrite query."""
        #relevance_score = state.get("relevance_score", 0)
        retry_count = state.get("retry_count", 0)
        
        #print(f"\n---DECISION: Score={relevance_score:.2f}, Retries={retry_count}---")
        print(f"\n---DECISION: Retries={retry_count}---")
        
        # If documents are relevant enough OR max retries reached, generate
        # if relevance_score >= settings.relevance_threshold or retry_count >= settings.max_retries:
        #     print("✅ Proceeding to generation")
        #     return "generate"
        # else:
        #     print("🔄 Rewriting query and retrying")
        #     return "rewrite"
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        #workflow.add_node("grade", self.grade_documents)
        workflow.add_node("rewrite", self.rewrite_query)
        workflow.add_node("generate", self.generate_answer)
        
        # Build graph
        workflow.set_entry_point("retrieve")
        #workflow.add_edge("retrieve", "grade")
        workflow.add_edge("retrieve", "generate")
        # Conditional edge: grade -> rewrite OR generate
        # workflow.add_conditional_edges(
        #     "grade",
        #     self.decide_to_generate,
        #     {
        #         "rewrite": "rewrite",
        #         "generate": "generate"
        #     }
        # )
        
        # After rewrite, go back to retrieve
        workflow.add_edge("rewrite", "retrieve")
        
        # After generate, end
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    

    def query(self, question: str, short_ctx: Optional[dict] = None, is_followup: bool = False) -> dict:
        short_ctx = short_ctx or {}
    
        def infer_brand_from_text(t: str) -> Optional[str]:
            t = (t or "").lower()
            brand_keywords = {
                "square": "Square",
                "clover": "Clover",
                "oracle": "Oracle",
                "micros": "Oracle",
                "lucas": "Oracle",
                "metos": "Metos",
                "la marzocco": "La Marzocco",
                "vulcan": "Vulcan",
                "lincoln": "Lincoln",
                "pitco": "Pitco",
                "manitowoc": "Manitowoc",
                "v400m": "V400m",
                "adyen": "V400m",
            }
            for k, v in brand_keywords.items():
                if k in t:
                    return v
            return None
    
        brand_to_type = {
            "Square": "POS",
            "Clover": "POS",
            "Oracle": "POS",
            "V400m": "POS",
            "Metos": "Coffee_Maker",
            "La Marzocco": "Espresso_Machine",
            "Vulcan": "Oven",
            "Lincoln": "Pizza_Oven",
            "Pitco": "Fryer",
            "Manitowoc": "Ice_Machine",
        }
    
        # ----------------------------
        # 1) Compute locks for follow-ups
        # ----------------------------
        locked_brand = None
        locked_type = None
        locked_title = None
    
        if is_followup:
            locked_brand = short_ctx.get("last_brand") or None
            locked_type = short_ctx.get("last_equipment_type") or None
            locked_title = short_ctx.get("last_manual_title") or None
    
            # If missing, infer from last_question text
            last_q = (short_ctx.get("last_question") or "").strip()
            if not locked_brand and last_q:
                locked_brand = infer_brand_from_text(last_q)
    
            if not locked_type and locked_brand:
                locked_type = brand_to_type.get(locked_brand)
    
        # ----------------------------
        # 2) Retrieval query augmentation for follow-ups
        # ----------------------------
        retrieval_query = question
        if is_followup:
            last_q = (short_ctx.get("last_question") or "").strip()
            if last_q:
                retrieval_query = f"{last_q}\nFollow-up: {question}"
    
        initial_state = {
            "question": question,                 # generation uses this
            "rewritten_question": retrieval_query, # retrieval uses this
            "generation": "",
            "documents": [],
            "retry_count": 0,
            "short_ctx": short_ctx,
    
            "locked_brand": locked_brand if is_followup else None,
            "locked_type": locked_type if is_followup else None,
            "locked_title": locked_title if is_followup else None,
    
            "primary_manual_title": None,
            "primary_equipment_type": None,
            "primary_equipment_brand": None,
        }
    
        result = self.workflow.invoke(initial_state)
    
        return {
            "question": question,
            "answer": result.get("generation", ""),
            "documents": result.get("documents", []),
            "retries": result.get("retry_count", 0),
            "primary_manual_title": result.get("primary_manual_title"),
            "primary_equipment_type": result.get("primary_equipment_type"),
            "primary_equipment_brand": result.get("primary_equipment_brand"),
        }
    
    
    

    def _pick_primary_manual(self, documents: List[dict]) -> dict:
        """
        Choose a single 'primary manual' from retrieved chunks.
        Simple rule: the manual title with the highest chunk count.
        Tie-break: highest max score.
        """
        if not documents:
            return {"title": None, "equipment_type": None, "equipment_brand": None}

        by_title = {}
        for d in documents:
            meta = d.get("metadata", {}) or {}
            title = meta.get("title", d.get("title", "Unknown Manual"))
            equip_type = meta.get("equipment_type", d.get("equipment_type"))
            equip_brand = meta.get("equipment_brand", d.get("equipment_brand"))
            score = float(d.get("score", 0))

            if title not in by_title:
                by_title[title] = {
                    "count": 0,
                    "max_score": 0.0,
                    "equipment_type": equip_type,
                    "equipment_brand": equip_brand,
                }
            by_title[title]["count"] += 1
            by_title[title]["max_score"] = max(by_title[title]["max_score"], score)

        # Pick max by (count, max_score)
        best_title = sorted(
            by_title.items(),
            key=lambda kv: (kv[1]["count"], kv[1]["max_score"]),
            reverse=True,
        )[0][0]

        return {
            "title": best_title,
            "equipment_type": by_title[best_title].get("equipment_type"),
            "equipment_brand": by_title[best_title].get("equipment_brand"),
        }
