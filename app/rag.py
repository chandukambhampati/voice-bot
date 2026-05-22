import re
from typing import List, Dict, Tuple

# RAG Documents
ONCOLOGY_DOCS = [
    {
        "id": "dr_bharat_credentials",
        "title": "Dr. Bharat Patodiya's Credentials and Experience",
        "text": "Dr. Bharat Patodiya is a highly acclaimed, award-winning cancer specialist (oncologist) who has studied and completed advanced fellowships at prestigious European universities. In India, he has worked at premier medical institutions including Hinduja Hospital in Mumbai and AIG Hospital in Hyderabad. He is exceptionally well-networked among oncology experts globally and across India.",
        "keywords": ["doctor", "credentials", "experience", "education", "degree", "who", "bharat", "patodiya", "qualification", "hinduja", "aig", "mumbai", "hyderabad"]
    },
    {
        "id": "novel_treatments",
        "title": "Novel Cancer Treatments and Clinical Trials",
        "text": "Advanced Cancer Center under Dr. Bharat Patodiya provides access to novel targeted therapies, advanced immunotherapies, and clinical-trial medications that are recently approved in Europe and the US but not yet available at standard Indian cancer centers. This enables us to provide hope and treatment directions even when other centers have given up.",
        "keywords": ["treatment", "medicine", "immunotherapy", "targeted", "clinical trial", "novel", "new", "europe", "hope", "advanced", "chemo", "chemotherapy"]
    },
    {
        "id": "consultation_process",
        "title": "Online Consultation Process & Travel Saving",
        "text": "To save patients and families the physical strain and cost of traveling during active illness, Dr. Bharat Patodiya and his team conduct initial case reviews online. They review current reports, advise on necessary investigations that can be done locally, and coordinate with local doctors. Travel to the physical clinic is only recommended later if absolutely necessary for active intervention.",
        "keywords": ["consultation", "process", "online", "travel", "save", "visit", "how", "reports", "upload", "local"]
    },
    {
        "id": "premium_consultation",
        "title": "Premium Consultation Offer Details",
        "text": "The Premium Consultation is priced at ₹5,000. It is chosen by most of our patients and includes: no strict time limit/cap on the session, one free follow-up consultation, a complete digital recording of the discussion, and a 100% money-back guarantee if no meaningful, novel treatment direction is provided for the patient.",
        "keywords": ["premium", "price", "cost", "charge", "fees", "5000", "guarantee", "refund", "recording", "follow-up", "unlimited"]
    },
    {
        "id": "standard_consultation",
        "title": "Standard Consultation Details",
        "text": "The Standard Consultation is priced at ₹3,000. It is designed to assist families facing severe financial constraints. It includes: a strict 30-minute time cap, no free follow-up sessions, no recording of the consultation, and does NOT include the 100% money-back guarantee.",
        "keywords": ["standard", "basic", "cheap", "cost", "price", "charge", "fees", "3000", "financial", "limit"]
    },
    {
        "id": "payment_booking",
        "title": "Payment Verification and Booking Logistics",
        "text": "Consultations are officially booked once the payment is completed. Callers will receive a QR code, Razorpay payment link, or official website booking page via SMS/WhatsApp. To confirm, the caller must share a screenshot of the completed payment. Reassurance is provided post-payment, and logistics coordinates reports and timeslots.",
        "keywords": ["book", "payment", "link", "qr", "pay", "screenshot", "razorpay", "upi", "gpay", "phonepe", "confirm"]
    },
    {
        "id": "breast_cancer_symptoms",
        "title": "Breast Cancer Symptoms and Support",
        "text": "Common symptoms of breast cancer include a painless lump or thickening in the breast, breast skin dimpling, swelling, changes in the nipple (like retraction or discharge), or localized pain. Dr. Bharat specializes in breast oncology and offers advanced therapies targeting HER2+ and triple-negative breast cancers.",
        "keywords": ["breast", "lump", "dimpling", "nipple", "swelling", "mastectomy", "her2", "mammography"]
    },
    {
        "id": "lung_cancer_symptoms",
        "title": "Lung Cancer Symptoms and Support",
        "text": "Key symptoms of lung cancer include a persistent cough that worsens, coughing up blood, chest pain, breathlessness or shortness of breath, wheezing, and unexplained weight loss. Dr. Bharat provides access to advanced targeted therapies for EGFR and ALK mutations, as well as immunotherapy.",
        "keywords": ["lung", "cough", "blood", "breath", "breathlessness", "chest", "egfr", "alk", "smoking"]
    },
    {
        "id": "colon_cancer_symptoms",
        "title": "Colon and Colorectal Cancer Symptoms",
        "text": "Symptoms of colon cancer include persistent changes in bowel habits (constipation, diarrhea), blood in the stool or rectal bleeding, persistent abdominal pain or cramping, vomiting, and feeling that the bowel does not empty completely. Special treatments include robotic surgery and advanced targeted monoclonal antibodies.",
        "keywords": ["colon", "colorectal", "stool", "bleeding", "rectal", "bowel", "constipation", "cramping", "stomach", "vomit"]
    }
]

class OncologyRAG:
    def __init__(self, docs: List[Dict] = ONCOLOGY_DOCS):
        self.docs = docs

    def retrieve(self, query: str, limit: int = 2) -> List[Dict]:
        """
        Simple lexical score based on matching keywords and query terms.
        """
        # Normalize query words
        query_words = set(re.findall(r'\w+', query.lower()))
        if not query_words:
            return []
            
        scored_docs = []
        for doc in self.docs:
            score = 0
            # Check keywords matches
            for keyword in doc.get("keywords", []):
                if keyword in query_words:
                    score += 3  # Higher weight for keyword hits
                    
            # Check content matches
            text_lower = doc["text"].lower()
            for word in query_words:
                if len(word) > 2 and word in text_lower:
                    score += 1
            
            if score > 0:
                scored_docs.append((score, doc))
                
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N docs
        return [doc for score, doc in scored_docs[:limit]]

# Simple test function
if __name__ == "__main__":
    rag = OncologyRAG()
    results = rag.retrieve("What is the cost of premium consultation with Dr. Bharat?")
    for doc in results:
        print(f"[{doc['id']}] {doc['title']}: {doc['text']}")
