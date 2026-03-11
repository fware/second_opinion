import streamlit as st
from streamlit_back_camera_input import back_camera_input
from google import genai
from google.genai.errors import ClientError
from fpdf import FPDF
import pandas as pd
import PIL.Image
import base64
import json
import time
import os
import difflib  # used for fuzzy string comparisons
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pypdf import PdfReader


# --- Configuration & Setup ---
st.set_page_config(page_title="Independent Auto Service: Second Opinion", page_icon="🚗", layout="centered")

# --- UI: White-Label CSS ---
hide_streamlit_style = """
    <style>
    /* Hides the top-right hamburger menu */
    #MainMenu {visibility: hidden !important;}
    
    /* Hides the 'Made with Streamlit' footer */
    footer {visibility: hidden !important;}
    
    /* Hides the top header bar completely */
    [data-testid="stHeader"] {display: none !important;}
    
    /* Hides the 'Hosted with Streamlit' teardrop badge */
    [data-testid="viewerBadge"] {display: none !important;}
    .viewerBadge_container {display: none !important;}
    
    /* Hides the 'Manage App' or 'Deploy' floating buttons */
    [data-testid="stAppDeployButton"] {display: none !important;}
    .stDeployButton {display: none !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# ---------------------------


# --- Mock Database for Independent Shop Pricing ---
@st.cache_data
def load_pricing_db():
    """
    Loads pricing from a local CSV. 
    Streamlit will automatically reload this if pricing.csv is modified!
    """
    try:
        df = pd.read_csv("pricing/pricing.csv", skipinitialspace=True)

        # Clean up any accidental spaces in the column names
        # df.columns = df.columns.str.strip()

        # We lower() the keys here so your fuzzy matching stays fast
        return dict(zip(df['Service Name'].str.lower(), df['Price']))
    except FileNotFoundError:
        st.error("Pricing database file not found!")
        return {}

# Replace your hardcoded MOCK_PRICING_DB with this call:
MOCK_PRICING_DB = load_pricing_db()

# --- Sidebar: Admin Controls & Cache Management ---
with st.sidebar:
    st.header("⚙️ Admin Dashboard")
    st.write("Update shop pricing database:")
    
    # 1. CSV Uploader
    new_pricing_file = st.file_uploader("Upload New Pricing (CSV)", type=["csv"])
    
    if new_pricing_file is not None:
        try:
            # Read and clean the uploaded CSV (handles the space-after-comma issue automatically)
            new_df = pd.read_csv(new_pricing_file, skipinitialspace=True)
            new_df.columns = new_df.columns.str.strip()
            
            if 'Service Name' in new_df.columns and 'Price' in new_df.columns:
                # Overwrite the local file
                new_df.to_csv("pricing.csv", index=False)
                st.success("✅ Pricing database updated live!")
                # Clear cache so the app reloads the new CSV immediately
                st.cache_data.clear() 
            else:
                st.error("CSV must contain 'Service Name' and 'Price' columns.")
        except Exception as e:
            st.error(f"Failed to process file: {e}")
            
    st.markdown("---")
    
    # 2. Display Active Pricing
    st.write("**Current Active Pricing:**")
    if MOCK_PRICING_DB:
        # Convert dictionary to DataFrame for a clean visual table
        display_df = pd.DataFrame(list(MOCK_PRICING_DB.items()), columns=["Service", "Price"])
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${float(x):.2f}")
        st.dataframe(display_df, hide_index=True, width='stretch')

    st.markdown("---")
    
    # 3. Clear Cache Button
    st.write("**Testing & Debugging:**")
    if st.button("🗑️ Clear Active Estimates", width='stretch'):
        st.session_state.estimate_cache = {}
        st.cache_data.clear()
        st.success("Cache cleared! The next upload will force a fresh AI analysis.")

# --- Streamlit UI ---
st.title("🚗 Independent Auto Service: Second Opinion")
st.markdown("Upload your  estimate to see if we can beat their price.")

# --- Core Functions ---
def extract_text_from_pdf(input_file):
    """Extracts raw text from the uploaded PDF."""
    reader = PdfReader(input_file)
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


def create_pdf_report(vehicle_name, comparison_results, total_dealer, total_independent, savings):
    """Generates a PDF estimate and returns it as raw bytes for Streamlit to download."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Independent Auto - Estimate Comparison", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(200, 10, txt=f"Vehicle: {vehicle_name}", ln=True, align='C')
    pdf.ln(10) # Add a line break
    
    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(90, 10, "Service", border=1)
    pdf.cell(40, 10, "Dealer Quote", border=1, align='C')
    pdf.cell(40, 10, "Our Estimate", border=1, align='C')
    pdf.ln()
    
    # Table Rows
    pdf.set_font("Arial", '', 10)
    for row in comparison_results:
        # Truncate long service names so they don't break the PDF table
        service_text = row['Service'][:45] + "..." if len(row['Service']) > 45 else row['Service']
        pdf.cell(90, 10, service_text, border=1)
        pdf.cell(40, 10, str(row['Dealer Quote']), border=1, align='C')
        pdf.cell(40, 10, str(row['Independent Estimate']), border=1, align='C')
        pdf.ln()
        
    pdf.ln(10)
    
    # Totals & Savings
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Dealership Total: ${total_dealer:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Independent Auto Total: ${total_independent:.2f}", ln=True)
    
    if savings > 0:
        pdf.set_text_color(0, 128, 0) # Green text for savings
        pdf.cell(200, 10, txt=f"Estimated Savings: ${savings:.2f}", ln=True)
        
    # Output the PDF as a byte string (required for Streamlit's download button)
    return pdf.output(dest='S').encode('latin-1')    

def send_sms_alert(to_phone_number, message_body):
    """
    Sends an SMS notification using Twilio.
    Returns True if successful, False if it fails.
    """
    try:
        # Pull credentials from Streamlit secrets
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        from_phone = st.secrets["TWILIO_PHONE_NUMBER"]

        # Initialize the Twilio client
        client = Client(account_sid, auth_token)

        # Send the message
        message = client.messages.create(
            body=message_body,
            from_=from_phone,
            to=to_phone_number
        )
        
        # Log the success for debugging
        print(f"DEBUG: SMS sent successfully! Message SID: {message.sid}")
        return True

    except Exception as e:
        print(f"DEBUG: Failed to send SMS. Error: {e}")
        return False

def send_email_alert(customer_name, customer_phone, alert_msg):
    """
    Sends an email notification using Python's built-in smtplib.
    Returns True if successful, False if it fails.
    """
    try:
        sender_email = st.secrets["SMTP_EMAIL"]
        sender_password = st.secrets["SMTP_PASSWORD"]
        receiver_email = st.secrets["RECEIVER_EMAIL"]

        # 1. Construct the Email structure
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"🚨 New Lead: Independent Auto Estimate - {customer_name}"

        # 2. Build the body of the email
        body = f"""
        You have a new lead from the Second Opinion Engine!
        
        Customer Name: {customer_name}
        Phone Number: {customer_phone}
        
        Details:
        {alert_msg}
        """
        msg.attach(MIMEText(body, 'plain'))

        # 3. Connect to the SMTP server (Gmail uses port 587)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls() # Upgrade the connection to a secure TLS encrypted connection
        server.login(sender_email, sender_password)
        
        # 4. Send and close
        server.send_message(msg)
        server.quit()
        
        print("DEBUG: Email sent successfully!")
        return True

    except Exception as e:
        print(f"DEBUG: Failed to send Email. Error: {e}")
        return False
        
# --- Utility Helpers ---
def service_matches_with_score(db_service: str, service_name: str, threshold: float = 0.5):
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


# --- UI: Input Selection ---
tab1, tab2 = st.tabs(["📁 Upload File", "📸 Take a Photo"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload Dealership Estimate (PDF or Photo)", 
        type=["pdf", "jpg", "jpeg", "png"]
    )

with tab2:
    #camera_file = st.camera_input("Take a clear picture of the estimate")
    st.write("Take a clear picture of the estimate:")
    # This custom widget forces the rear camera by default
    camera_file = back_camera_input()



# Unify the input so the rest of your app doesn't have to change!
active_file = uploaded_file or camera_file


# --- Initialize Streamlit Multi-File Memory ---
if "estimate_cache" not in st.session_state:
    # Create an empty dictionary to hold all processed files
    st.session_state.estimate_cache = {}

if active_file is not None:
    # --- NEW: Normalize the file properties based on the input source ---
    if hasattr(active_file, 'name'):
        # It came from the standard file uploader
        file_name = active_file.name
        file_type = active_file.type
        file_bytes = active_file.getvalue()
    else:
        # It came from the custom camera widget (raw BytesIO stream)
        file_bytes = active_file.getvalue()
        
        # Hash the image bytes to create a stable, unique file name for the cache!
        import hashlib
        file_hash = hashlib.md5(file_bytes).hexdigest()
        file_name = f"camera_capture_{file_hash}.jpg"
        file_type = "image/jpeg"

    # --- UI: File Preview ---
    if file_type != "application/pdf":
        # It's an image
        st.image(file_bytes, caption="Uploaded Estimate", use_container_width=True) 
    else:
        # It's a PDF
        st.write("**Uploaded Estimate:**")
        # Read the file bytes and encode to base64
        base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
        # Embed the PDF using HTML
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    # ------------------------

    # 1. Check if we have already processed THIS specific file today
    if file_name in st.session_state.estimate_cache:
        st.success(f"⚡ Loaded estimate from cache! (Saved an API call)")
        estimate_data = st.session_state.estimate_cache[file_name]

    # 2. If it's a new file, call the AI and save it to the dictionary
    else:
        with st.spinner("Extracting current estimate data and calculating our price..."):
            # Notice we pass file_bytes and file_type here now!
            estimate_data = parse_document_with_llm_v2(file_bytes, file_type)
            
            # Save it to memory so we never process this exact file again this session
            if estimate_data:
                st.session_state.estimate_cache[file_name] = estimate_data  
                
    # 3. Proceed normally!                  
    if estimate_data:
        st.success(f"Estimate parsed successfully for: **{estimate_data.get('vehicle', 'Unknown Vehicle')}**")
            
        # 3. Build Comparison Data
        comparison_results = []
        total_dealer = 0
        total_independent = 0
         
        with st.status("Comparing prices with Independent Auto Service database...", expanded=True) as status:
            for item in estimate_data.get("repairs", []):
                service_name = item.get("service", "").lower()
                dealer_price = item.get("quoted_price", 0.0)
                # st.write(f"Processing: {service_name.title()} - Dealer Quote: ${dealer_price:.2f}") # Debug line to show each service being processed


                best_score = 0
                independent_price = "Custom Quote Needed"
                confidence_label = "No Match 🔴"                
                # Check EVERY item in the database, don't just stop at the first one
                for db_service, db_price in MOCK_PRICING_DB.items():
                    is_match, conf_score = service_matches_with_score(db_service, service_name, threshold=0.5)
                    
                    # Convert your emoji labels into a mathematical weight
                    weight_map = {"High 🟢": 3, "Good 🟡": 2, "Partial 🟠": 1, "None 🔴": 0}
                    current_weight = weight_map.get(conf_score, 0)
                    
                    # If this match is BETTER than the last one we found, overwrite it
                    if is_match and current_weight > best_score:
                        best_score = current_weight
                        independent_price = db_price
                        confidence_label = conf_score
                        
                        # If we hit a perfect 3, we don't need to keep searching
                        if best_score == 3:
                            break
                            
                # Add the absolute BEST result we found to the total
                if isinstance(independent_price, (int, float)):
                    total_independent += independent_price
                    total_dealer += dealer_price    # I want to only total services when we also have an independent estimate

                
                comparison_results.append({
                    "Service": service_name.title(),
                    "Dealer Quote": f"${dealer_price:.2f}" if dealer_price > 0 else "Unpriced",
                    "Independent Estimate": f"${independent_price:.2f}" if isinstance(independent_price, (int, float)) else independent_price,
                    "Match Confidence": confidence_label
                })
            status.update(label="Comparison Complete!", state="complete", expanded=True)
            
            # 4. Display Results
            st.subheader("Cost Comparison")
            df = pd.DataFrame(comparison_results)
            st.dataframe(df, width='stretch')
            
            # 5. The "Hook" (Call to Action & Error Handling)
            st.markdown("---")
            
            # Initialize variables to safely handle the lead capture form
            show_form = False
            alert_msg_template = ""
            
            # Scenario A: We found repairs, but no dealership prices
            if len(estimate_data.get("repairs", [])) > 0 and total_dealer == 0:
                st.warning("⚠️ **No Dealership Prices Detected**")
                st.write("We successfully read the recommended repairs, but we didn't find any quoted prices on the document you uploaded.")
                
                if total_independent > 0:
                    st.success(f"Good news! We went ahead and pulled Independent Auto Service's estimates anyway. We estimate this work will cost around **${total_independent:.2f}**.")
                    show_form = True
                    alert_msg_template = "New Lead! {name} ({phone}) uploaded an unpriced estimate for a {vehicle}. Independent estimate: ${total_independent:.2f}."
                else:
                    st.info("Independent Auto Service will need to provide a custom quote for these specific items.")
                    show_form = True
                    alert_msg_template = "New Lead! {name} ({phone}) needs a custom quote for a {vehicle} (unpriced upload)."

            # Scenario B: Standard Comparison & Savings
            elif total_independent > 0 and total_dealer > total_independent:
                savings = total_dealer - total_independent
                st.success(f"🎉 We estimate we can save you around **${savings:.2f}** on these repairs!")
                show_form = True
                alert_msg_template = "New Lead! {name} ({phone}) wants a second opinion on a {vehicle}. Estimated savings: ${savings:.2f}."
                
                # --- NEW: PDF Download Button ---
                st.write("---")
                st.write("**Save a copy of this estimate for your records:**")
                
                # Generate the PDF bytes
                vehicle_name = estimate_data.get('vehicle', 'Unknown Vehicle')
                pdf_bytes = create_pdf_report(vehicle_name, comparison_results, total_dealer, total_independent, savings)
                
                # Create the Streamlit download button
                st.download_button(
                    label="📄 Download Estimate as PDF",
                    data=pdf_bytes,
                    file_name=f"Independent_Auto_Estimate_{vehicle_name.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
                st.write("---")

            # Scenario C: Dealership is somehow cheaper or exactly the same
            elif total_independent > 0 and total_dealer > 0 and total_dealer <= total_independent:
                st.info("It looks like the dealership quote is highly competitive for these specific services.")
                show_form = True
                alert_msg_template = "New Lead! {name} ({phone}) checked prices for a {vehicle}. Dealer quote was competitive."


            # --- Lead Capture Form ---
            if show_form:
                st.subheader("Lock in this Estimate / Get a Custom Quote")
                
                with st.form("lead_capture_form"):
                    st.write("Enter your details and Independent Auto Service will reach out to confirm your appointment.")
                    
                    customer_name = st.text_input("Full Name")
                    customer_phone = st.text_input("Phone Number (for SMS)")
                    
                    # The form submit button handles the actual action (removed the loose button you had at the end)
                    submitted = st.form_submit_button("Send to Independent Auto Service", type="primary")
                    
                    if submitted:
                        if customer_name and customer_phone:
                            # 1. Grab the vehicle safely
                            vehicle_name = estimate_data.get('vehicle', 'Unknown Vehicle')
                            
                            # 2. Format the SMS string based on which scenario triggered the form
                            if "savings" in alert_msg_template:
                                alert_msg = alert_msg_template.format(name=customer_name, phone=customer_phone, vehicle=vehicle_name, savings=(total_dealer - total_independent))
                            elif "total_independent" in alert_msg_template:
                                alert_msg = alert_msg_template.format(name=customer_name, phone=customer_phone, vehicle=vehicle_name, total_independent=total_independent)
                            else:
                                alert_msg = alert_msg_template.format(name=customer_name, phone=customer_phone, vehicle=vehicle_name)
                            
                            # 3. Trigger the notification (Make sure you have the send_sms_alert function defined in your script!)
                            # Replace the hardcoded number with your verified test number for the demo!

                            demo_phone_number = "+14695582111"

                            with st.spinner("Sending lead to the shop..."):
                                # send_sms_alert(demo_phone_number, alert_msg)
                                email_success = send_email_alert(customer_name, customer_phone, alert_msg)

                            if email_success:                           
                                st.success("Success! Independent Auto Service has received your estimate. They will text or call you shortly.")
                            else:
                                st.warning("We saved your information, but the SMS alert system is currently offline.")
                        else:
                            st.warning("Please provide your name and phone number.")

            st.button("Book an Appointment to Lock in this Estimate", type="primary")
