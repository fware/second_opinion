import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd
import PIL.Image
import json
import time
import os
import difflib  # used for fuzzy string comparisons

# --- Configuration & Setup ---
st.set_page_config(page_title="Rallye Auto: Second Opinion", page_icon="🚗", layout="centered")

# In production, use st.secrets. For local testing, you can set it here or in your env.
API_KEY = st.secrets.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# --- Mock Database for Independent Shop Pricing ---
# In a real app, this would be a Supabase/Postgres query based on make/model/service.
MOCK_PRICING_DB = {
    "Rear Shock Absorber Replacement": 449.99,
    "brake pads replacement": 219.99,
    "12v battery replacement": 179.99,
    "synthetic oil change": 75.00,
    "wiper blades - front": 24.99,
    "four wheel alignment": 129.99,
    "Alternator Replacement": 514.99,
    "Spark Plugs Replacement": 109.99,
    "Starter Motor Replacement": 359.99,
    "Front Brake Replacement": 179.99,
    "Rear Brake Replacement": 189.99,
    "AC Compressor Replacement": 699.99,
    "Clutch Assembly Replacement": 2899.99,
}

# --- Core Functions ---
def extract_text_from_pdf(uploaded_file):
    """Extracts raw text from the uploaded PDF."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def parse_estimate_with_llm(raw_text):
    """Uses Gen AI to convert unstructured PDF text into structured JSON."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are an expert automotive service advisor. I am providing you with the raw text extracted from a car repair estimate PDF.
    Your job is to identify the vehicle make and model, and list the recommended repairs along with their quoted prices.
    
    Return ONLY a valid JSON object with the following structure:
    {{
        "vehicle": "Make and Model (e.g., Lexus CT200h)",
        "repairs": [
            {{"service": "Name of service", "quoted_price": 0.00}}
        ]
    }}
    
    Raw text from estimate:
    {raw_text}
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean up the response to ensure it's pure JSON (removing markdown blocks if present)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(cleaned_response)
    except Exception as e:
        st.error(f"Error parsing estimate: {e}")
        return None


def parse_document_with_llm_v2(uploaded_file):
    """
    Parses a PDF or Image estimate and returns a structured Python Dictionary.
    Guarantees 'vehicle' and 'repairs' keys are present and correctly populated.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # 1. Strict prompt engineering to enforce the schema and rules
    prompt = """
    You are an expert Automotive Service Estimator. Extract the vehicle information and repair line items from the provided document.
    
    RULES:
    1. Identify the Year, Make, and Model of the vehicle. Combine them into a single string (e.g., "2015 Lexus CT200h"). 
       If you cannot find this information, you MUST use the exact string: "Unknown Vehicle".
    2. Combine separated labor and parts into a single service item if they belong together.
    3. Ensure quoted_price is always a number/float (e.g., 150.00) without currency symbols.

    You MUST return ONLY a valid JSON object matching this exact structure, with no markdown formatting or extra text:
    {
        "vehicle": "Year Make Model",
        "repairs": [
            {
                "service": "Clear name of the repair",
                "quoted_price": 0.00
            }
        ]
    }
    """

    try:
        # 2. Route based on file type
        if uploaded_file.type == "application/pdf":
            # Handle PDF (Extract text first using your helper)
            raw_text = extract_text_from_pdf(uploaded_file)
            response = model.generate_content([prompt, raw_text])
        else:
            # Handle Image (Pass the image object directly)
            img = PIL.Image.open(uploaded_file)
            response = model.generate_content([prompt, img])
            
        # 3. Clean the response string from the LLM
        raw_json_string = response.text.strip()
        
        # Remove markdown code blocks if the LLM ignores instructions
        if raw_json_string.startswith("```json"):
            raw_json_string = raw_json_string[7:]
        if raw_json_string.endswith("```"):
            raw_json_string = raw_json_string[:-3]
            
        # 4. Convert to Python Dictionary
        parsed_dict = json.loads(raw_json_string.strip())
        
        # 5. SAFETY CHECK: Ensure we have a valid dictionary and inject defaults if the LLM hallucinated
        if not isinstance(parsed_dict, dict):
            parsed_dict = {}

        # If vehicle key is missing, or the value is empty/null, force it to Unknown Vehicle
        if "vehicle" not in parsed_dict or not parsed_dict["vehicle"]:
            parsed_dict["vehicle"] = "Unknown Vehicle"
            
        # Ensure repairs is always a list, even if empty
        if "repairs" not in parsed_dict:
            parsed_dict["repairs"] = []
            
        return parsed_dict

    except json.JSONDecodeError as e:
        st.error("Failed to parse AI response. The document might be too blurry or unreadable.")
        # Print the raw text to terminal so you can debug why it failed
        print(f"DEBUG: Raw AI Output causing JSON error:\n{response.text}") 
        return None
    except Exception as e:
        st.error(f"An error occurred during document processing: {e}")
        return None
    

def parse_sophisticated_estimate_with_llm(raw_text):
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are an expert Automotive Service Estimator. Your goal is to parse raw text from a dealership repair estimate and normalize it for a comparison tool.

    ### RULES:
    1. GROUPING: If labor and parts for the same repair (e.g., 'Brake Pads' and 'Brake Labor') are listed separately, COMBINE them into a single service item.
    2. CLEANING: Remove internal dealership codes (e.g., 'OP CODE 02LEXZ').
    3. PRICING: Sum the labor and parts for the combined service. Ensure the price is a float.
    4. UNCERTAINTY: If a line item is a "Recommendation" but has no price, include it with a price of 0.0.

    ### OUTPUT FORMAT:
    Return ONLY a valid JSON object:
    {{
        "vehicle": "Year, Make, and Model",
        "repairs": [
            {{
                "service": "Clear, concise name of the repair",
                "total_dealer_price": 0.00,
                "is_bundled": true/false (true if you combined parts and labor)
            }}
        ]
    }}

    ### RAW TEXT:
    {raw_text}
    """
    
    try:
        response = model.generate_content(prompt)
        # Handle cases where LLM might wrap response in markdown code blocks
        json_str = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Logic Error: {e}")
        return None
    

# --- Utility Helpers ---
def service_matches_with_score(db_service: str, service_name: str, threshold: float = 0.6):
    """Returns a tuple: (is_match, confidence_string)"""
    db = db_service.lower()
    name = service_name.lower()
    
    # 1. High Confidence: Exact or substring match
    if db == name or db in name or name in db:
        return True, "High 🟢"

    # 2. Good Confidence: Word overlap
    db_words = set(db.split())
    name_words = set(name.split())
    if len(db_words & name_words) >= 2:
        return True, "Good 🟡"

    # 3. Partial Confidence: Fuzzy difflib ratio
    if difflib.SequenceMatcher(None, db, name).ratio() >= threshold:
        return True, "Partial 🟠"

    return False, "None 🔴"



def service_matches(db_service: str, service_name: str, threshold: float = 0.6) -> bool:
    """Return True when two service descriptions appear to refer to the same repair.

    Matching strategy:
    1. case‑insensitive substring check (existing behaviour)
    2. word overlap – at least two words in common
    3. fuzzy ratio from difflib (default threshold 0.6, tune as needed)

    This handles examples like
        db_service="wiper blades front"
        service_name="Front Wiper Blade Replacement"
    because they share the words "wiper" and "front" and the overall similarity is high.
    """
    db = db_service.lower()
    name = service_name.lower()
    if db in name or name in db:
        return True

    db_words = set(db.split())
    name_words = set(name.split())
    if len(db_words & name_words) >= 2:
        return True

    if difflib.SequenceMatcher(None, db, name).ratio() >= threshold:
        return True

    return False


# --- Streamlit UI ---
st.title("🚗 Rallye Auto: Second Opinion Engine")
st.markdown("Upload your  estimate to see if we can beat their price.")

# Update the file uploader to accept images
uploaded_file = st.file_uploader(
    "Upload Dealership Estimate (PDF or Photo)", 
    type=["pdf", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Preview the image if it's a photo
    if uploaded_file.type != "application/pdf":
        st.image(uploaded_file, caption="Uploaded Estimate", use_container_width=True)

    with st.spinner("Extracting Work current estimate data and calculating our price..."):
        # 1. Extract Text
        # raw_text = extract_text_from_pdf(uploaded_file)
        
        # 2. Parse with LLM
        estimate_data = parse_document_with_llm_v2(uploaded_file)
        
        if estimate_data:
            st.success(f"Estimate parsed successfully for: **{estimate_data.get('vehicle', 'Unknown Vehicle')}**")
            
            # 3. Build Comparison Data
            comparison_results = []
            total_dealer = 0
            total_rallye = 0
           
            with st.status("Comparing prices with Rallye Auto database...", expanded=True) as status:
                for item in estimate_data.get("repairs", []):
                    service_name = item.get("service", "").lower()
                    dealer_price = item.get("quoted_price", 0.0)
                    # st.write(f"Processing: {service_name.title()} - Dealer Quote: ${dealer_price:.2f}") # Debug line to show each service being processed
                    total_dealer += dealer_price
                    
                    # Basic matching logic against our mock DB, now with improved partial/fuzzy matching
                    confidence_label = "No Match 🔴" # Default
                    rallye_price = "Custom Quote Needed"
                    
                    for db_service, db_price in MOCK_PRICING_DB.items():
                        is_match, conf_score = service_matches_with_score(db_service, service_name)
                        if is_match:
                            rallye_price = db_price
                            total_rallye += rallye_price
                            confidence_label = conf_score
                            break   

                    comparison_results.append({
                        "Service": service_name.title(),
                        "Dealer Quote": f"${dealer_price:.2f}" if isinstance(dealer_price, (int, float)) else dealer_price,
                        "Rallye Estimate": f"${rallye_price:.2f}" if isinstance(rallye_price, (int, float)) else rallye_price,
                        "Match Confidence": confidence_label # Add the confidence label to the results for display
                    })
                status.update(label="Comparison Complete!", state="complete", expanded=True)
            
            # 4. Display Results
            st.subheader("Cost Comparison")
            df = pd.DataFrame(comparison_results)
            st.dataframe(df, use_container_width=True)
            
            # 5. The "Hook" (Call to Action)
            if total_rallye > 0 and total_dealer > total_rallye:
                savings = total_dealer - total_rallye
                st.info(f"🎉 We estimate we can save you around **${savings:.2f}** on these repairs.")
                
                st.markdown("---")
                st.subheader("Lock in this Estimate")
                
                # Use a Streamlit form to capture user details
                with st.form("lead_capture_form"):
                    st.write("Enter your details and Rallye Auto will reach out to confirm your appointment.")
                    
                    customer_name = st.text_input("Full Name")
                    customer_phone = st.text_input("Phone Number (for SMS)")
                    
                    # The form submit button
                    submitted = st.form_submit_button("Send to Rallye Auto", type="primary")
                    
                    if submitted:
                        if customer_name and customer_phone:
                            # Construct the alert message
                            alert_msg = f"New Lead! {customer_name} ({customer_phone}) wants a second opinion on a {estimate_data.get('vehicle')}. Estimated savings: ${savings:.2f}."
                            
                            # Trigger the notification (function defined below)
                            send_sms_alert("+14695582111", alert_msg) # Replace with Rallye's actual phone number
                            
                            st.success("Success! Rallye Auto has received your estimate. They will text or call you shortly.")
                        else:
                            st.warning("Please provide your name and phone number.")


            st.button("Book an Appointment to Lock in this Estimate", type="primary")