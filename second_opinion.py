import streamlit as st
from google import genai
from google.genai.errors import ClientError
import pandas as pd
import PIL.Image
import json
import time
import os
import difflib  # used for fuzzy string comparisons
import io
from pypdf import PdfReader


# --- Configuration & Setup ---
st.set_page_config(page_title="Independent Auto Service: Second Opinion", page_icon="🚗", layout="centered")

# --- Mock Database for Independent Shop Pricing ---
@st.cache_data
def load_pricing_db():
    """
    Loads pricing from a local CSV. 
    Streamlit will automatically reload this if pricing.csv is modified!
    """
    try:
        df = pd.read_csv("pricing.csv", skipinitialspace=True)

        # Clean up any accidental spaces in the column names
        # df.columns = df.columns.str.strip()

        # We lower() the keys here so your fuzzy matching stays fast
        return dict(zip(df['Service Name'].str.lower(), df['Price']))
    except FileNotFoundError:
        st.error("Pricing database file not found!")
        return {}

# Replace your hardcoded MOCK_PRICING_DB with this call:
MOCK_PRICING_DB = load_pricing_db()
# In a real app, this would be a Supabase/Postgres query based on make/model/service.
# MOCK_PRICING_DB = {
#     "Rear Shock Absorber Replacement": 449.99,
#     "brake pads replacement": 219.99,
#     "12v battery replacement": 179.99,
#     "synthetic oil change": 75.00,
#     "wiper blades - front": 24.99,
#     "four wheel alignment": 129.99,
#     "Alternator Replacement": 514.99,
#     "Spark Plugs Replacement": 109.99,
#     "Starter Motor Replacement": 359.99,
#     "Front Brake Replacement": 179.99,
#     "Rear Brake Replacement": 189.99,
#     "AC Compressor Replacement": 699.99,
#     "Clutch Assembly Replacement": 2899.99,
# }

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

@st.cache_data(show_spinner=False)
def parse_document_with_llm_v2(file_bytes, file_type):
    """
    Parses a PDF or Image estimate and returns a structured Python Dictionary.
    Updated to use the new google-genai SDK.
    """
    # 1. Initialize the new Client
    client = genai.Client(api_key=st.secrets.get("GEMINI_API_KEY"))
    
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

    max_retries = 3
    base_delay = 2 

    for attempt in range(max_retries):
        try:
            # 2. Route based on file type using the new client syntax
            if file_type == "application/pdf":
                # Assuming extract_text_from_pdf can accept raw bytes via io.BytesIO
                import io
                raw_text = extract_text_from_pdf(io.BytesIO(file_bytes))
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, raw_text]
                )
            else:
                import io
                import PIL.Image
                img = PIL.Image.open(io.BytesIO(file_bytes))
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, img]
                )
            
            break # Break out of loop if successful
            
        except ClientError as e:
            # 3. Catch the new 429 Rate Limit Error
            if e.code == 429:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt) 
                    st.warning(f"⏳ API rate limit hit. Pausing for {sleep_time} seconds before retrying... (Attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(sleep_time)
                else:
                    st.error("🚨 API Quota completely exceeded. Please try again tomorrow or upgrade your Gemini API plan.")
                    return None
            else:
                st.error(f"API Error: {e.message}")
                return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None
    
    # 4. JSON parsing logic stays exactly the same!
    try:
        raw_json_string = response.text.strip()
        
        if raw_json_string.startswith("```json"):
            raw_json_string = raw_json_string[7:]
        if raw_json_string.endswith("```"):
            raw_json_string = raw_json_string[:-3]
            
        import json
        parsed_dict = json.loads(raw_json_string.strip())
        
        if not isinstance(parsed_dict, dict):
            parsed_dict = {}

        if "vehicle" not in parsed_dict or not parsed_dict["vehicle"]:
            parsed_dict["vehicle"] = "Unknown Vehicle"
            
        if "repairs" not in parsed_dict:
            parsed_dict["repairs"] = []
            
        return parsed_dict

    except Exception as e:
        st.error("Failed to parse AI response. The document might be too blurry or unreadable.")
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
st.title("🚗 Independent Auto: Second Opinion Engine")
st.markdown("Upload your  estimate to see if we can beat their price.")

# Update the file uploader to accept images
uploaded_file = st.file_uploader(
    "Upload Dealership Estimate (PDF or Photo)", 
    type=["pdf", "jpg", "jpeg", "png"]
)

# --- Initialize Streamlit Multi-File Memory ---
if "estimate_cache" not in st.session_state:
    # Create an empty dictionary to hold all processed files
    st.session_state.estimate_cache = {}

if uploaded_file is not None:
    file_name = uploaded_file.name
    
    # Preview the image if it's a photo
    if uploaded_file.type != "application/pdf":
        st.image(uploaded_file, caption="Uploaded Estimate", use_container_width=True)

    # 1. Check if we have already processed THIS specific file today
    if file_name in st.session_state.estimate_cache:
        st.success(f"⚡ Loaded {file_name} from cache! (Saved an API call)")
        estimate_data = st.session_state.estimate_cache[file_name]

    # 2. If it's a new file, call the AI and save it to the dictionary
    else:
        with st.spinner("Extracting current estimate data and calculating our price..."):
            file_bytes = uploaded_file.getvalue()
            estimate_data = parse_document_with_llm_v2(file_bytes, uploaded_file.type)
            
            # Save it to memory so we never process this filename again this session
            if estimate_data:
                st.session_state.estimate_cache[file_name] = estimate_data  

    # 3. Proceed normally!                  
    if estimate_data:
        st.success(f"Estimate parsed successfully for: **{estimate_data.get('vehicle', 'Unknown Vehicle')}**")
            
        # 3. Build Comparison Data
        comparison_results = []
        total_dealer = 0
        total_Independent = 0
         
        with st.status("Comparing prices with Independent Auto database...", expanded=True) as status:
            for item in estimate_data.get("repairs", []):
                service_name = item.get("service", "").lower()
                dealer_price = item.get("quoted_price", 0.0)
                # st.write(f"Processing: {service_name.title()} - Dealer Quote: ${dealer_price:.2f}") # Debug line to show each service being processed
                total_dealer += dealer_price
                  
                # Basic matching logic against our mock DB, now with improved partial/fuzzy matching
                confidence_label = "No Match 🔴" # Default
                Independent_price = "Custom Quote Needed"
                    
                for db_service, db_price in MOCK_PRICING_DB.items():
                    is_match, conf_score = service_matches_with_score(db_service, service_name)
                    if is_match:
                        Independent_price = db_price
                        total_Independent += Independent_price
                        confidence_label = conf_score
                        break   

                comparison_results.append({
                    "Service": service_name.title(),
                    "Dealer Quote": f"${dealer_price:.2f}" if isinstance(dealer_price, (int, float)) else dealer_price,
                    "Independent Estimate": f"${Independent_price:.2f}" if isinstance(Independent_price, (int, float)) else Independent_price,
                    "Match Confidence": confidence_label # Add the confidence label to the results for display
                })
            status.update(label="Comparison Complete!", state="complete", expanded=True)
            
# 4. Display Results
            st.subheader("Cost Comparison")
            df = pd.DataFrame(comparison_results)
            st.dataframe(df, use_container_width=True)
            
            # 5. The "Hook" (Call to Action & Error Handling)
            st.markdown("---")
            
            # Initialize variables to safely handle the lead capture form
            show_form = False
            alert_msg_template = ""
            
            # Scenario A: We found repairs, but no dealership prices
            if len(estimate_data.get("repairs", [])) > 0 and total_dealer == 0:
                st.warning("⚠️ **No Dealership Prices Detected**")
                st.write("We successfully read the recommended repairs, but we didn't find any quoted prices on the document you uploaded.")
                
                if total_Independent > 0:
                    st.success(f"Good news! We went ahead and pulled Independent Auto's estimates anyway. We estimate this work will cost around **${total_Independent:.2f}**.")
                    show_form = True
                    alert_msg_template = "New Lead! {name} ({phone}) uploaded an unpriced estimate for a {vehicle}. Independent estimate: ${Independent_total:.2f}."
                else:
                    st.info("Independent Auto will need to provide a custom quote for these specific items.")
                    show_form = True
                    alert_msg_template = "New Lead! {name} ({phone}) needs a custom quote for a {vehicle} (unpriced upload)."

            # Scenario B: Standard Comparison & Savings
            elif total_Independent > 0 and total_dealer > total_Independent:
                savings = total_dealer - total_Independent
                st.success(f"🎉 We estimate we can save you around **${savings:.2f}** on these repairs!")
                show_form = True
                alert_msg_template = "New Lead! {name} ({phone}) wants a second opinion on a {vehicle}. Estimated savings: ${savings:.2f}."

            # Scenario C: Dealership is somehow cheaper or exactly the same
            elif total_Independent > 0 and total_dealer > 0 and total_dealer <= total_Independent:
                st.info("It looks like the dealership quote is highly competitive for these specific services.")
                show_form = True
                alert_msg_template = "New Lead! {name} ({phone}) checked prices for a {vehicle}. Dealer quote was competitive."

            # --- Lead Capture Form ---
            if show_form:
                st.subheader("Lock in this Estimate / Get a Custom Quote")
                
                with st.form("lead_capture_form"):
                    st.write("Enter your details and Independent Auto will reach out to confirm your appointment.")
                    
                    customer_name = st.text_input("Full Name")
                    customer_phone = st.text_input("Phone Number (for SMS)")
                    
                    # The form submit button handles the actual action (removed the loose button you had at the end)
                    submitted = st.form_submit_button("Send to Independent Auto", type="primary")
                    
                    if submitted:
                        if customer_name and customer_phone:
                            # 1. Grab the vehicle safely
                            vehicle_name = estimate_data.get('vehicle', 'Unknown Vehicle')
                            
                            # 2. Format the SMS string based on which scenario triggered the form
                            if "savings" in alert_msg_template:
                                alert_msg = alert_msg_template.format(name=customer_name, phone=customer_phone, vehicle=vehicle_name, savings=(total_dealer - total_Independent))
                            elif "Independent_total" in alert_msg_template:
                                alert_msg = alert_msg_template.format(name=customer_name, phone=customer_phone, vehicle=vehicle_name, Independent_total=total_Independent)
                            else:
                                alert_msg = alert_msg_template.format(name=customer_name, phone=customer_phone, vehicle=vehicle_name)
                            
                            # 3. Trigger the notification (Make sure you have the send_sms_alert function defined in your script!)
                            # send_sms_alert("+14695582111", alert_msg) 
                            
                            st.success("Success! Independent Auto has received your estimate. They will text or call you shortly.")
                        else:
                            st.warning("Please provide your name and phone number.")

            st.button("Book an Appointment to Lock in this Estimate", type="primary")