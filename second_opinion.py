import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd
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
    "Spark Plug Replacement": 199.99,
    "Starter Motor Replacement": 359.99,
    "Front Brake Replacement": 179.99,
    "Rear Brake Replacement": 189.99,
    "AC Compressor Replacement": 699.99,
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
st.markdown("Upload your dealership estimate to see if we can beat their price.")

uploaded_file = st.file_uploader("Upload Estimate (PDF)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting dealership data and calculating our price..."):
        # 1. Extract Text
        raw_text = extract_text_from_pdf(uploaded_file)
        
        # 2. Parse with LLM
        estimate_data = parse_estimate_with_llm(raw_text)
        
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
                    st.write(f"Processing: {service_name.title()} - Dealer Quote: ${dealer_price:.2f}")
                    total_dealer += dealer_price
                    
                    # Basic matching logic against our mock DB, now with improved partial/fuzzy matching
                    rallye_price = "Custom Quote Needed"
                    for db_service, db_price in MOCK_PRICING_DB.items():
                        st.write(f"Checking against DB service: {db_service.title()} - Price: ${db_price:.2f}")
                        if service_matches(db_service, service_name):
                            st.write(f"!!!MATCH FOUND!!!: {db_service.title()} - Price: ${db_price:.2f}")
                            rallye_price = db_price
                            total_rallye += rallye_price
                            break
                    
                    comparison_results.append({
                        "Service": service_name.title(),
                        "Dealer Quote": f"${dealer_price:.2f}" if isinstance(dealer_price, (int, float)) else dealer_price,
                        "Rallye Estimate": f"${rallye_price:.2f}" if isinstance(rallye_price, (int, float)) else rallye_price
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