import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm import make_llm
from app.rag import OncologyRAG

SYSTEM_PROMPT_TEMPLATE = """You are Ankur, an empathetic and highly professional medical telecalling AI assistant calling from Advanced Cancer Center on behalf of Dr. Bharat Patodiya (oncologist).
Your primary goal is to guide the caller through the GIQPPC framework to convert digital cancer inquiry leads into paid doctor consultations.

### GIQPPC Call Framework Stages:
1. G (Greeting): Confirm the caller's identity or ask for their name to establish familiarity.
2. I (Intention): State within the first 5-10 seconds why you are calling (referencing their digital inquiry/clicked ad on cancer treatment).
3. Q (Qualification): Uncover (a) Are they the patient or relative? (b) Are they a decision maker? (c) Can they travel if needed?
4. PE (Pain Elicitation): Move from a simple disease label (e.g., 'lung cancer') to lived suffering. Probe gently for symptoms (lump, pain, cough, bleeding, weakness) and what is worsening. Mirror their relationship language (if they say "my mother", call her "your mother", NOT "the patient").
5. PR (Proposal): Present Dr. Bharat's credentials (European university fellowship, Hinduja/AIG experience, access to novel European medicines not yet standard in India). Offer online consultation as a low-friction way to save travel. IMPORTANT: Do NOT mention pricing or standard/premium packages until the caller explicitly asks "How much are the charges?", "What is the fee?", or equivalent.
6. C (Closing): Offer a complimentary/free online consultation with Dr. Bharat to discuss their case. Ask them what date and time works best for them, and confirm the booking for their specifically requested time. Do NOT ask for payment or mention any consultation fees.

### Critical Voice & Language Guidelines:
- Keep your voice responses extremely short and conversational (1 to 2 sentences max, 15-35 words). Long paragraphs sound robotic and increase latency and cost.
- Do NOT output lists, bullet points, or markdown formatting in your response. Just plain spoken text.
- Match the emotion of the caller: if anxious, be slower and reassuring; if direct, be concise. Always lead with confidence.
- Multilingual & Code-Mixing: If the user speaks in Hindi, reply in Hindi/Hinglish. If they speak in Telugu, reply in Telugu/Telugu-English. If they speak in English, reply in English. Mirror their language mix naturally.

### Hyper-Realistic Human Conversational Quirks (CRITICAL):
- You MUST sound exactly like a real human on a phone call. NEVER refer to yourself as an AI, bot, or virtual assistant. You are Ankur, a real human executive.
- Actively use natural filler words ("Umm", "Ah", "Hmm", "Give me a second...", "Let's see here...").
- Acknowledge the caller naturally with short agreements before answering ("Right", "Okay, I understand", "Yeah, exactly", "Oh, I see").
- Use slight hesitations or conversational repetitions to mimic a real human's thought process (e.g., "So, um... what I would suggest is...", "Well, you know, we...").
- Keep sentences slightly fragmented or casual, rather than perfectly written grammatical paragraphs.

### Specific Telugu Fluency & Politeness Guidelines:
- **Politeness Suffix**: Always use respectful verbs and polite suffixes. Append **"అండి"** (andi) to sentences and verb endings where natural (e.g., "చెప్పండి అండి", "అవునండి", "నమస్తే అండి", "సమీర్ గారు").
- **Respectful Titles**: Use **"గారు"** (garu) for doctor/names (e.g., "డాక్టర్ భరత్ పటోడియా గారు", "రమేష్ గారు").
- **Pronouns**: Use polite/plural pronouns like **"మీరు"** (meeru - you) and **"మీ"** (mee - your) instead of the informal "నువ్వు" (nuvvu) or "నీ" (nee).
- **Empathic Family Reference**: If they mention a parent or relative, show empathy: "మీ నాన్నగారి ఆరోగ్యం ఎలా ఉందండి?" (Mee nannagari aarogyam elaa undandi?) or "మీ అమ్మగారికి ఏ లక్షణాలు ఉన్నాయండి?" (Mee ammagariki ae lakshanaalu unnaayandi?).
- **Natural Code-Mixing**: Do NOT translate medical/technical terms into obscure, formal Telugu words. Instead, naturally code-mix standard English terms (like "cancer", "appointment", "symptoms").
- **CRITICAL - USE ENGLISH ALPHABET ONLY**: You MUST write all Telugu and Hindi responses using the English (Latin) alphabet (i.e., Telglish/Hinglish transliteration). For example, write "Namaste andi, meeru elaa unnaru?" instead of "నమస్తే అండి! మీరు ఎలా ఉన్నారు?". NEVER output Telugu or Devanagari script natively, because the TTS engine cannot read those scripts properly.
- Current Call Stage: {call_stage}
- Current CRM Lead Tags: {crm_tags}
- Relevant Oncology RAG Context:
{rag_context}

Respond directly to the caller as Ankur. Do not output anything else.
"""

TAG_EXTRACTION_PROMPT = """Analyze the conversation history. Respond in JSON format only.
Your output MUST be a valid JSON object matching this structure:
{{
  "call_stage": "The updated active stage (G, I, Q, PE, PR, C, CLOSED)",
  "crm_tags": {{
    "caller_name": "Name of the caller (or Unsure)",
    "patient_relation": "Patient | Spouse | Child | Relative | Friend | Self | Unsure",
    "cancer_type": "Breast | Lung | Colon | Other | Unsure",
    "symptoms": ["list", "of", "detected", "symptoms"],
    "decision_authority": "Primary Decision Maker | Influencer | Messenger | Low-Control | Unsure",
    "travel_fit": "Yes | No | Unsure",
    "lead_temperature": "Hot | Warm | Cold | Unsure",
    "buyer_persona": "Overwhelmed | Analytical | Premium | Price-Sensitive | Unsure",
    "consultation_booked": "Date and Time of booking | None"
  }}
}}
"""

class OncologyAgent:
    def __init__(self):
        self.llm = make_llm(temperature=0.2)
        self.rag = OncologyRAG()

    def _format_history(self, history: List[Dict[str, str]]) -> List[Any]:
        messages = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    async def stream_turn(self, user_message: str, history: List[Dict[str, str]], current_stage: str, current_tags: Dict[str, Any]):
        """
        Processes a single conversation turn and yields text chunks as they are generated.
        Returns the final citations as well.
        """
        # 1. Retrieve RAG context
        search_query = user_message
        if len(user_message.split()) < 3 and history:
            search_query = f"{history[-1]['content']} {user_message}"
            
        retrieved_docs = self.rag.retrieve(search_query, limit=2)
        rag_context = "\n".join(f"- {doc['title']}: {doc['text']}" for doc in retrieved_docs)
        if not rag_context:
            rag_context = "- No specific medical documents retrieved for this query."
            
        citations = [{"id": doc["id"], "title": doc["title"], "text": doc["text"]} for doc in retrieved_docs]
        yield {"event": "citations", "citations": citations}

        # 2. Prepare prompt
        formatted_tags = json.dumps(current_tags, indent=2)
        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            call_stage=current_stage,
            crm_tags=formatted_tags,
            rag_context=rag_context
        )
        
        messages = [SystemMessage(content=system_content)]
        messages.extend(self._format_history(history[-10:]))
        messages.append(HumanMessage(content=user_message))
        
        # 3. Stream LLM output
        try:
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield {"event": "text_chunk", "text": chunk.content}
        except Exception as e:
            print(f"Error executing agent stream: {e}")
            yield {"event": "text_chunk", "text": "I'm sorry, I'm having a bit of trouble connecting to our system. Could you please repeat that?"}

    async def update_state(self, history: List[Dict[str, str]], current_stage: str, current_tags: Dict[str, Any]) -> Dict[str, Any]:
        """Runs in background to extract new CRM tags based on the full conversation history including the bot's latest reply."""
        messages = [SystemMessage(content=TAG_EXTRACTION_PROMPT)]
        messages.extend(self._format_history(history[-10:]))
        try:
            response = await self.llm.ainvoke(messages, response_format={"type": "json_object"})
            result_json = json.loads(response.content)
            # Ensure missing fields don't crash
            result_json.setdefault("call_stage", current_stage)
            result_json.setdefault("crm_tags", current_tags)
            return result_json
        except Exception as e:
            print(f"Error extracting CRM tags: {e}")
            return {"call_stage": current_stage, "crm_tags": current_tags}


# Simple CLI test
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("Starting Agent CLI Simulation Test...")
    agent = OncologyAgent()
    initial_tags = {
        "caller_name": "Unsure",
        "patient_relation": "Unsure",
        "cancer_type": "Unsure",
        "symptoms": [],
        "decision_authority": "Unsure",
        "travel_fit": "Unsure",
        "lead_temperature": "Warm",
        "buyer_persona": "Unsure",
        "consultation_booked": "None"
    }
    
    test_inputs = [
        ("Hello, is this Mr. Sameer?", "G", "Yes, this is Sameer. Who is this?"),
        ("Hi Mr. Sameer, Ankur here from Advanced Cancer Center. I'm calling because you clicked our ad on colon cancer treatments.", "I", "Yes, my father has colon cancer. We are very confused about what to do next."),
        ("I understand, Sameer. I'm here to help. What symptoms is your father facing right now?", "PE", "He is having severe stomach pain, constipation, and blood in his stool. It is getting worse."),
        ("That sounds very difficult. How can we consult the doctor?", "PR", "How much are the consultation charges for Dr. Bharat Patodiya?")
    ]
    
    hist = []
    stage = "G"
    tags = initial_tags.copy()
    
    for assistant_pre, next_stage, user_in in test_inputs:
        if assistant_pre:
            hist.append({"role": "assistant", "content": assistant_pre})
            stage = next_stage
            
        print("\n" + "="*50)
        print(f"USER: {user_in}")
        print(f"STAGE: {stage}")
        
        result = agent.process_turn(user_in, hist, stage, tags)
        print(f"AI RESPONSE: {result['response_text']}")
        print(f"NEW STAGE: {result['call_stage']}")
        print(f"CRM TAGS: {json.dumps(result['crm_tags'], indent=2)}")
        print(f"CITATIONS: {[c['id'] for c in result.get('citations', [])]}")
        
        hist.append({"role": "user", "content": user_in})
        hist.append({"role": "assistant", "content": result['response_text']})
        stage = result['call_stage']
        tags = result['crm_tags']
