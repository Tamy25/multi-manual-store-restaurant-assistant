"""
Author: Tamasa Patra
Purpose: Streamlit chat interface for Multi-Manual RAG System
FIXED: Follow-up detection now respects explicit equipment/brand mentions
"""

import streamlit as st
from src.langgraph_workflow import CoffeeMakerRAG
from src.chroma_client import ChromaDBManager
import time

import re
from typing import Dict, Any, Optional, Tuple


# =============================================================================
# FIX #1: Enhanced follow-up detection that checks for NEW equipment mentions
# =============================================================================

# Brand keywords map (same as in langgraph_workflow.py - keep in sync!)
BRAND_KEYWORDS = {
    "square": "Square",
    "clover": "Clover",
    "oracle": "Oracle",
    "micros": "Oracle",
    "lucas": "Oracle",
    "metos": "Metos",
    "vulcan": "Vulcan",
    "lincoln": "Lincoln",
    "pitco": "Pitco",
    "la marzocco": "La Marzocco",
    "manitowoc": "Manitowoc",
    "v400m": "V400m",
    "adyen": "V400m",
}

# Equipment type keywords
EQUIPMENT_TYPE_KEYWORDS = {
    "POS": ["pos", "terminal", "payment", "refund", "void","transaction", "totals", "receipt", "merchant", "card", "paper roll"],
    "Coffee_Maker": ["coffee maker", "coffee machine", "brew", "descale", "carafe"],
    "Espresso_Machine": ["espresso", "steam wand", "portafilter"],
    "Fryer": ["fryer", "fry", "oil", "basket", "boil out", "boil-out", "filtering"],
    "Pizza_Oven": ["pizza oven", "impinger"],
    "Oven": ["oven", "convection", "bake", "broil", "thermostat", "roast"],
    "Ice_Machine": ["ice machine", "ice maker"],
}


def detect_equipment_in_query(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect if query explicitly mentions a brand or equipment type.
    
    Returns:
        (detected_brand, detected_equipment_type) - either can be None
    """
    t = text.lower()
    
    detected_brand = None
    detected_type = None
    
    # Check for brand mentions
    for keyword, brand in BRAND_KEYWORDS.items():
        if keyword in t:
            detected_brand = brand
            break
    
    # Check for equipment type mentions
    for equip_type, keywords in EQUIPMENT_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                detected_type = equip_type
                break
        if detected_type:
            break
    
    return detected_brand, detected_type


def is_followup_message(text: str, short_ctx: Dict[str, Any]) -> bool:
    """
    Enhanced follow-up detector.
    
    KEY FIX: If the user explicitly mentions a DIFFERENT equipment type or brand
    than what's in the context, this is NOT a follow-up — it's a new question.
    
    Returns True for short confirmations/status updates that depend on previous context.
    Returns False if the user is asking about different equipment.
    """
    if not text:
        return False

    t = text.strip().lower()
    
    # ==========================================================================
    # FIX: Check if user is asking about DIFFERENT equipment than last context
    # ==========================================================================
    detected_brand, detected_type = detect_equipment_in_query(text)
    
    last_brand = short_ctx.get("last_brand", "")
    last_type = short_ctx.get("last_equipment_type", "")
    
    # If user explicitly mentions a brand that's different from context → NOT a follow-up
    if detected_brand and last_brand and detected_brand != last_brand:
        print(f"🔄 NEW EQUIPMENT DETECTED: Brand changed from {last_brand} to {detected_brand}")
        return False
    
    # If user explicitly mentions an equipment type different from context → NOT a follow-up
    if detected_type and last_type and detected_type != last_type:
        print(f"🔄 NEW EQUIPMENT DETECTED: Type changed from {last_type} to {detected_type}")
        return False
    
    # If user mentions ANY brand/type explicitly and there was no prior context → NOT a follow-up
    if (detected_brand or detected_type) and (not last_brand and not last_type):
        return False
    
    # ==========================================================================
    # Original follow-up heuristics (for actual follow-ups)
    # ==========================================================================
    
    # Super short replies are usually follow-ups
    if len(t.split()) <= 6:
        return True

    # Common follow-up patterns
    followup_patterns = [
        r"^(yes|no|yeah|yep|nope|ok|okay|done|cool|sure)\b",
        r"^(it|this|that|he|she|they)\b",
        r"\b(it is|it's|it's)\b",
        r"\b(power(ed)?\s+on|power(ed)?\s+off|turn(ed)?\s+on|turn(ed)?\s+off)\b",
        r"\b(blinking|flashing|solid)\b",
        r"\b(not\s+connected|connected)\b",
        r"\b(i did|i tried|i can't|i cannot|i can't|i see|i don't|i do not)\b",
        r"\b(error|code|message)\b",
    ]
    
    # Only consider it a follow-up if it matches patterns AND doesn't mention new equipment
    if any(re.search(p, t) for p in followup_patterns):
        # But only if no explicit equipment change detected
        if not detected_brand and not detected_type:
            return True
    
    return False


def extract_primary_manual_title(docs: list) -> Optional[str]:
    """
    Picks the manual title that appears most often among retrieved docs.
    """
    if not docs:
        return None

    counts = {}
    for d in docs:
        meta = d.get("metadata", {}) or {}
        title = meta.get("title", "Unknown Manual")
        counts[title] = counts.get(title, 0) + 1

    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else None


# Page config
st.set_page_config(
    page_title="Store Manual Assistant",
    page_icon="src/assets/shop.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #7F8C8D;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "short_ctx" not in st.session_state:
    st.session_state.short_ctx = {
        "last_question": "",
        "last_answer": "",
        "last_manual_title": "",
        "last_equipment_type": "",
        "last_brand": ""
    }

if "last_followup_options" not in st.session_state:
    st.session_state.last_followup_options = []

if "rag_system" not in st.session_state:
    with st.spinner("🔧 Initializing Multi-Manual RAG system..."):
        try:
            st.session_state.rag_system = CoffeeMakerRAG()
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"❌ Error initializing system: {e}")
            st.session_state.initialized = False

# Sidebar
with st.sidebar:
    st.markdown("# 🏪 Store Manual Assistant")
    st.markdown("*One assistant. All manuals. Zero guesswork.*")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.short_ctx = {
            "last_question": "",
            "last_answer": "",
            "last_manual_title": "",
            "last_equipment_type": "",
            "last_brand": "",
        }
        st.rerun()

# Main content
st.markdown('<p class="main-header">🏪 Store Manual Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Welcome! 👋 I can help you quickly find answers from our equipment manuals — just ask what you need.</p>',
            unsafe_allow_html=True)

if not st.session_state.get("initialized", False):
    st.warning("⚠️ System not initialized. Please run setup first:")
    st.code("python main.py setup", language="bash")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask about any restaurant equipment...")

# Process the prompt
if user_input:
    

    # =========================================================================
    # EXPAND NUMBERED INPUT BEFORE CALLING RAG
    # =========================================================================
    original_input = user_input  # Keep original for display
        
        # Check if user typed just a number (1, 2, 3, 4, 5)
    if user_input.strip() in ['1', '2', '3', '4', '5']:
        options = st.session_state.get('last_followup_options', [])
        idx = int(user_input.strip()) - 1
            
        if 0 <= idx < len(options):
            user_input = options[idx]  # EXPAND: "2" → "Resetting the dry-boil protection"
            print(f"📝 Expanded '{original_input}' → '{user_input}'")
    # ==========================================================================
    st.session_state.messages.append({"role": "user", "content": original_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            try:
                # =============================================================
                # Pass short_ctx to is_followup_message for comparison
                # =============================================================
                has_prev = bool(st.session_state.short_ctx.get("last_question"))
                followup = has_prev and is_followup_message(user_input, st.session_state.short_ctx)
                
                # Debug output (visible in Streamlit)
                # detected_brand, detected_type = detect_equipment_in_query(user_input)
                # st.write(f"🔍 Debug: followup={followup}, detected_brand={detected_brand}, detected_type={detected_type}")
                # st.write(f"📋 Context: last_brand={st.session_state.short_ctx.get('last_brand')}, last_type={st.session_state.short_ctx.get('last_equipment_type')}")
                
                result = st.session_state.rag_system.query(
                    user_input,
                    short_ctx=st.session_state.short_ctx,
                    is_followup=followup,
                )
                response = result["answer"]

                # =========================================================================
                # EXTRACT AND STORE FOLLOW-UP OPTIONS FOR NEXT TURN
                # =========================================================================
                options = re.findall(r'(\d+)\.\s*([^\n]+)', response)
                st.session_state.last_followup_options = [opt[1].strip() for opt in options]
                print(f"📋 Stored {len(st.session_state.last_followup_options)} options: {st.session_state.last_followup_options}")
                # =========================================================================
                
                docs = result.get("documents", []) or []
                primary_manual = extract_primary_manual_title(docs) or ""
                
                # Update context for next turn
                st.session_state.short_ctx = {
                    "last_question": user_input,
                    "last_answer": response,
                    "last_manual_title": result.get("primary_manual_title") or primary_manual,
                    "last_equipment_type": result.get("primary_equipment_type") or "",
                    "last_brand": result.get("primary_equipment_brand") or "",
                }
                
                response_time = time.time() - start_time
                
                # # # Display response with typing effect
                # # full_response = ""
                # # for chunk in response.split():
                # #     full_response += chunk + " "
                # #     message_placeholder.markdown(full_response + "▌")
                # #     time.sleep(0.02)

                # Show loading, then full response
                message_placeholder.markdown("*Generating response...*")
                time.sleep(0.3)  # Brief pause
                message_placeholder.markdown(response)
                
                message_placeholder.markdown(response)
                
                # Calculate distributions for metadata
                equipment_dist = {}
                manual_dist = {}
                
                for doc in result.get("documents", []):
                    equip_type = doc.get('equipment_type', 'unknown')
                    equipment_dist[equip_type] = equipment_dist.get(equip_type, 0) + 1
                    
                    metadata = doc.get('metadata', {})
                    title = metadata.get('title', 'Unknown Manual')
                    manual_dist[title] = manual_dist.get(title, 0) + 1
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "metadata": {
                        "num_docs": len(result.get("documents", [])),
                        "time": response_time,
                        "retries": result.get("retries", 0),
                        "equipment_dist": equipment_dist,
                        "manual_dist": manual_dist
                    }
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
