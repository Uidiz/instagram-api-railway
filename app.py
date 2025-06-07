import os
import re
import json
import time
import functools # For the API key decorator
import requests # For AssemblyAI and potentially other HTTP calls
import yt_dlp # For downloading Instagram videos
import google.generativeai as genai # For Gemini LLM
import cv2 # For frame extraction
from concurrent.futures import ThreadPoolExecutor # For potential parallel tasks
import threading

from flask import Flask, Response, request, jsonify
from flask_cors import CORS

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for all routes
app.config['JSON_SORT_KEYS'] = False # Preserve order in JSON responses

# --- Configuration & API Keys (Read from Environment Variables) ---
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
RAILWAY_API_KEY = os.environ.get("RAILWAY_API_KEY") # API Key for securing this Flask API

BATCH_PROCESSING_RESULTS = {}

# --- Basic Configuration Validation ---
if not ASSEMBLYAI_API_KEY:
    print("CRITICAL WARNING: ASSEMBLYAI_API_KEY environment variable not set. Transcription will fail.")
if GEMINI_API_KEY:
    if "\n" in GEMINI_API_KEY or "\r" in GEMINI_API_KEY or " " in GEMINI_API_KEY:
        print(f"CRITICAL WARNING: GEMINI_API_KEY Invalid Metadata : {repr(GEMINI_API_KEY)}")
    else:
        print(f"GEMINI_API_KEY is loaded correctly: {GEMINI_API_KEY[:6]}...")
else:
    print("CRITICAL WARNING: GEMINI_API_KEY environment variable not set. LLM processing will fail.")
if not RAILWAY_API_KEY:
    print("CRITICAL WARNING: RAILWAY_API_KEY environment variable not set. The API will be inaccessible or insecure.")

# --- Gemini API Setup ---
gemini_model_instance = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using a model capable of handling text and multiple images
        gemini_model_instance = genai.GenerativeModel(model_name="gemini-1.5-pro-latest") # or gemini-1.5-flash-latest
        print("Gemini API configured successfully with multi-modal model.")
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini API: {e}")
else:
    print("WARNING: Gemini API Key is missing. LLM features will fail.")


# --- Global Variables ---
LOGS = [] # List to store logs generated during a request - simple in-memory logging
FRAME_EXTRACTION_INTERVAL_SECONDS = 5

# --- Logging Helper ---
def log(msg):
    """Appends a message to the LOGS list and prints it."""
    print(msg) # Railway captures stdout/stderr
    LOGS.append(str(msg))

# --- API Key Authentication Decorator ---
def require_api_key(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not RAILWAY_API_KEY: # Allow access if not set (for local dev ease)
            log("API Key check skipped: RAILWAY_API_KEY not configured on server.")
        else:
            provided_key = request.headers.get('X-API-Key')
            if not provided_key or provided_key != RAILWAY_API_KEY:
                log(f"Forbidden: Invalid or missing API Key. Provided: '{provided_key}'")
                return jsonify({"error": "Forbidden: Invalid or missing API Key"}), 403
        return f(*args, **kwargs)
    return decorated_function

# -------------------------------------------------
# 1. Download Instagram video by URL using yt_dlp
# -------------------------------------------------
def download_instagram_video(instagram_url, output_dir='temp_videos'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Using a fixed name within a unique_id subfolder to simplify finding the file
    # Create a unique sub-directory for this download to avoid filename collisions
    unique_id = f"insta_{int(time.time())}_{os.urandom(4).hex()}"
    video_specific_download_dir = os.path.join(output_dir, unique_id)
    os.makedirs(video_specific_download_dir, exist_ok=True)
    
    output_filename_template = os.path.join(video_specific_download_dir, 'downloaded_video.%(ext)s')

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_filename_template,
        'quiet': False,
        'noprogress': True,
        'noplaylist': True,
        'merge_output_format': 'mp4', # Ensure merged output is mp4
    }
    downloaded_file_path = None
    try:
        log(f"Attempting to download video from URL: {instagram_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(instagram_url, download=True)
            
            # Try to find the downloaded file path
            # yt-dlp might name the file downloaded_video.mp4 or similar based on content
            if 'requested_downloads' in info_dict and info_dict['requested_downloads']:
                 downloaded_file_path = info_dict['requested_downloads'][0]['filepath']
            elif '_filename' in info_dict: # Fallback if using an older yt-dlp or different flow
                downloaded_file_path = info_dict['_filename']
            else: # If still not found, scan the specific directory for likely candidates
                found_files = [
                    os.path.join(video_specific_download_dir, f) 
                    for f in os.listdir(video_specific_download_dir) 
                    if f.startswith("downloaded_video.") and (f.endswith(".mp4") or f.endswith(".mkv"))
                ]
                if found_files:
                    downloaded_file_path = found_files[0] # Take the first match

            if downloaded_file_path and os.path.exists(downloaded_file_path):
                log(f"Video downloaded successfully: {downloaded_file_path}")
                return downloaded_file_path, video_specific_download_dir # Return the dir for cleanup
            else:
                log(f"Error: yt_dlp downloaded, but could not confirm final file path for {instagram_url} in {video_specific_download_dir}")
                return None, video_specific_download_dir


    except yt_dlp.utils.DownloadError as e:
        log(f"yt_dlp DownloadError for {instagram_url}: {e}")
        return None, video_specific_download_dir
    except Exception as e:
        log(f"Error downloading Instagram video '{instagram_url}': {e}")
        return None, video_specific_download_dir


# -------------------------------------------------
# 2. Transcribe audio using AssemblyAI
# -------------------------------------------------
def transcribe_audio_assemblyai(api_key, file_path):
    if not api_key:
        log("ERROR: AssemblyAI API Key is missing for transcription.")
        return None
    if not file_path or not os.path.exists(file_path):
        log(f"ERROR: File not found for transcription: {file_path}")
        return None

    log(f"Starting transcription with AssemblyAI for file: {file_path}")
    headers = {"authorization": api_key}
    try:
        with open(file_path, 'rb') as f:
            response_upload = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=f)
        response_upload.raise_for_status()
        upload_url = response_upload.json()['upload_url']
        log("File uploaded successfully to AssemblyAI.")

        transcript_request = {'audio_url': upload_url, 'auto_highlights': True}
        response_transcript_req = requests.post("https://api.assemblyai.com/v2/transcript", json=transcript_request, headers=headers)
        response_transcript_req.raise_for_status()
        transcript_id = response_transcript_req.json()['id']
        log(f"Transcription requested successfully. ID: {transcript_id}")

        transcript_endpoint_poll = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
        max_retries = 20
        retry_delay = 10
        for i in range(max_retries):
            response_poll = requests.get(transcript_endpoint_poll, headers=headers)
            response_poll.raise_for_status()
            result = response_poll.json()
            status = result.get('status')
            log(f"AssemblyAI transcription status ({i+1}/{max_retries}): {status}")
            if status == 'completed':
                log("Transcription completed by AssemblyAI.")
                return result.get('text', '')
            elif status == 'error':
                log(f"Transcription failed at AssemblyAI: {result.get('error')}")
                return None
            time.sleep(retry_delay)
        log("Max retries reached for AssemblyAI transcription.")
        return None
    except Exception as e:
        log(f"Error during AssemblyAI processing: {e}")
        return None

# -------------------------------------------------
# 3. Extract Frames from Video
# -------------------------------------------------
def extract_frames_from_video(video_path, interval_seconds=5):
    if not video_path or not os.path.exists(video_path):
        log(f"ERROR: Video file not found for frame extraction: {video_path}")
        return []

    log(f"Extracting frames from {video_path} every {interval_seconds} seconds.")
    frames_data = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"ERROR: OpenCV could not open video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Handle cases where FPS is not reported or is zero
        log(f"Warning: Video FPS reported as 0 for {video_path}. Frame extraction might be unreliable.")
        cap.release()
        return []

    frame_interval = int(fps * interval_seconds)
    if frame_interval == 0: frame_interval = 1 # Ensure at least some frames are processed if interval is too small for FPS

    frame_count = 0
    extracted_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video or error

        if frame_count % frame_interval == 0:
            ret_encode, buffer = cv2.imencode('.jpg', frame) # Encode frame to JPEG
            if ret_encode:
                frames_data.append(buffer.tobytes())
                extracted_count += 1
            else:
                log(f"Warning: Could not encode frame {frame_count} to JPEG.")
        frame_count += 1
    
    cap.release()
    log(f"Extracted {extracted_count} frames from video.")
    return frames_data

# -------------------------------------------------
# 4. Process Content (Transcript + Frames) with Gemini LLM
# -------------------------------------------------
def process_content_with_gemini(transcript_text, frames_list_bytes):
    if not gemini_model_instance:
        log("ERROR: Gemini model not initialized. Cannot process content.")
        return None

    log(f"Processing transcript and {len(frames_list_bytes)} frames with Gemini LLM...")

    # Construct the prompt for Gemini
    # The initial text part of the prompt
    text_prompt_part = f"""
You are given the transcript of an Instagram video and a sequence of image frames extracted from it approximately every {FRAME_EXTRACTION_INTERVAL_SECONDS} seconds.
The video is likely a recipe, a DIY guide, or a similar instructional content.

--- TRANSCRIPT START ---
{transcript_text if transcript_text and transcript_text.strip() else "No textual transcript was available or it was empty."}
--- TRANSCRIPT END ---

Following this text, there are {len(frames_list_bytes)} image frames provided from the video, in chronological order.

Your task is to analyze BOTH the transcript (if available) AND the visual information from these image frames to extract the following details.
Prioritize information that is clearly supported by either the text or the visuals, or both.
Provide your output as a single, valid JSON object with the specified keys.

JSON Output Structure:
1.  "title": A concise and relevant title for the content (e.g., recipe name, DIY project name). If unclear, derive a best-effort title.
2.  "summary": A brief summary of what the video is about (e.g., what is being made or demonstrated), incorporating visual cues if they add clarity.
3.  "instructions": An array of strings, where each string is a step in the recipe instructions or DIY guide. Use both text and visual information to make these steps as clear and accurate as possible. If no clear steps, provide an empty array.
4.  "ingredients": An array of objects, primarily for recipes. Each object should have two properties:
    * "item": The name of the ingredient (string).
    * "quantity": The quantity of the ingredient, preferably using the metric system (e.g., "100 g", "250 ml", "1 unit", "2 tsp"). If quantity is not clear or applicable, use a descriptive string, "as needed", or leave it as an empty string.
    Identify ingredients from the transcript and confirm or supplement with visual evidence from the frames if possible. If not a recipe or ingredients are not applicable, this array should be empty.

Example JSON output:
{{
  "title": "Delicious Homemade Pizza",
  "summary": "This video demonstrates how to make pizza from scratch, showing the dough preparation, adding toppings, and baking.",
  "instructions": [
    "Prepare pizza dough and let it rise (visual: dough rising in a bowl).",
    "Stretch out the dough on a floured surface.",
    "Spread tomato sauce evenly over the dough (visual: sauce being spread).",
    "Add cheese and your favorite toppings (visual: cheese, pepperoni, and peppers being added).",
    "Bake in a preheated oven at 220°C until crust is golden and cheese is bubbly."
  ],
  "ingredients": [
    {{"item": "Pizza dough", "quantity": "1 ball"}},
    {{"item": "Tomato sauce", "quantity": "150 ml"}},
    {{"item": "Mozzarella cheese", "quantity": "200 g"}},
    {{"item": "Pepperoni", "quantity": "50 g"}},
    {{"item": "Bell peppers", "quantity": "1/2 unit, sliced"}}
  ]
}}

If the content does not seem to be a recipe or structured guide, or if the information cannot be reliably extracted even with visual aid,
provide a generic title and summary, and empty arrays for instructions and ingredients.

Output ONLY the valid JSON object. Do not include any explanatory text, markdown formatting like ```json, or any other characters outside the JSON object itself.
"""

    # Create the list of parts for the Gemini API call
    prompt_parts = [text_prompt_part]
    for i, frame_bytes in enumerate(frames_list_bytes):
        prompt_parts.append({"mime_type": "image/jpeg", "data": frame_bytes})
        if i < len(frames_list_bytes) -1 : # Add a text separator/contextualizer if needed between images.
             prompt_parts.append(f"\n--- End of Frame {i+1} / Start of Frame {i+2} --- \n")


    try:
        # It's generally recommended to make one call with all parts for coherent understanding
        # rather than describing frames individually and then summarizing descriptions for this specific final JSON task.
        response = gemini_model_instance.generate_content(prompt_parts)
        
        # Attempt to clean and parse the response. Gemini can sometimes include markdown.
        cleaned_response_text = response.text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:] # Remove ```json
        elif cleaned_response_text.startswith("```"): # Catch if only ``` is present
            cleaned_response_text = cleaned_response_text[3:]
        
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3] # Remove trailing ```
        
        log(f"Gemini raw response (cleaned snippet): {cleaned_response_text[:500]}...")
        
        extracted_data = json.loads(cleaned_response_text)
        log("Successfully processed content with Gemini and parsed JSON.")
        return extracted_data
    except json.JSONDecodeError as e:
        log(f"ERROR: Failed to parse Gemini JSON response: {e}. Response was: {response.text if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        log(f"ERROR: An unexpected error occurred with Gemini LLM processing: {e}")
        # You might want to inspect response.prompt_feedback if available
        if 'response' in locals() and response.prompt_feedback:
            log(f"Gemini Prompt Feedback: {response.prompt_feedback}")
        return None

# -------------------------------------------------
# Main Orchestration Function for an Instagram Post
# -------------------------------------------------
def process_instagram_post_logic(instagram_url):
    log(f"--- Starting processing for URL: {instagram_url} ---")
    LOGS.clear()

    video_file_path = None
    video_download_dir = None # To store the unique directory for this download

    try:
        # 1. Download video
        video_file_path, video_download_dir = download_instagram_video(instagram_url, output_dir='temp_videos')
        if not video_file_path:
            return {"error": "Failed to download video from Instagram URL.", "details": LOGS}

        # 2. Extract Frames
        extracted_frames = extract_frames_from_video(video_file_path, interval_seconds=FRAME_EXTRACTION_INTERVAL_SECONDS)
        # No need to fail if no frames, LLM prompt can handle it.

        # 3. Transcribe audio
        transcript = transcribe_audio_assemblyai(ASSEMBLYAI_API_KEY, video_file_path)
        # No need to fail if no transcript, LLM prompt can handle it.
        if transcript is None:
            log("Transcription failed or returned None. Proceeding without transcript text.")
            transcript = "" # Ensure it's a string for the LLM prompt
        elif not transcript.strip():
            log("Transcription resulted in empty text. Proceeding with empty transcript.")
        
        # 4. Process content (transcript + frames) with LLM
        # Limit number of frames if it's excessive to avoid overly large API calls or costs
        # E.g., max 20-30 frames. For a 5s interval, this is ~1.5-2.5 minutes of video.
        # You might adjust this limit based on typical video length and API constraints.
        MAX_FRAMES_TO_LLM = 24 # Approx 2 minutes of frames at 5s interval
        if len(extracted_frames) > MAX_FRAMES_TO_LLM:
            log(f"Warning: Extracted {len(extracted_frames)} frames. Truncating to first {MAX_FRAMES_TO_LLM} for LLM.")
            frames_to_send_to_llm = extracted_frames[:MAX_FRAMES_TO_LLM]
        else:
            frames_to_send_to_llm = extracted_frames

        processed_data = process_content_with_gemini(transcript, frames_to_send_to_llm)
        if not processed_data:
            return {"error": "Failed to process content with LLM.", "details": LOGS}

        log(f"--- Successfully processed URL: {instagram_url} ---")
        return {"success": True, "data": processed_data, "logs": LOGS}

    except Exception as e:
        log(f"FATAL Orchestration Error for {instagram_url}: {e}")
        return {"error": "An unexpected error occurred during processing.", "details": str(e), "logs": LOGS}
    finally:
        # 5. Cleanup downloaded video file and its unique directory
        if video_download_dir and os.path.exists(video_download_dir):
            try:
                # Remove individual files first if any were left due to error before dir removal
                if video_file_path and os.path.exists(video_file_path):
                    os.remove(video_file_path)
                    log(f"Cleaned up temporary video file: {video_file_path}")
                
                # Attempt to remove the directory; will fail if not empty from other files.
                # A more robust cleanup would be shutil.rmtree(video_download_dir)
                import shutil
                shutil.rmtree(video_download_dir)
                log(f"Cleaned up temporary video directory: {video_download_dir}")
            except Exception as e:
                log(f"Error cleaning up temporary video directory {video_download_dir}: {e}")

# -------------------------------------------------
# Background Batch Processing Function
# -------------------------------------------------
def process_batch_in_background(batch_id, instagram_urls):
    """Process a batch of Instagram URLs in the background"""
    print(f"🚀 Starting background processing for batch {batch_id} with {len(instagram_urls)} URLs")
    
    # Initialize batch results
    BATCH_PROCESSING_RESULTS[batch_id] = {
        "status": "processing",
        "total": len(instagram_urls),
        "completed": 0,
        "failed": 0,
        "results": [],
        "start_time": time.time(),
        "last_updated": time.time()
    }
    
    try:
        for i, url in enumerate(instagram_urls):
            print(f"📝 Processing URL {i+1}/{len(instagram_urls)}: {url}")
            
            # Process individual URL
            result = process_instagram_post_logic(url)
            
            # Update batch results
            BATCH_PROCESSING_RESULTS[batch_id]["results"].append(result)
            BATCH_PROCESSING_RESULTS[batch_id]["last_updated"] = time.time()
            
            if result.get("success"):
                BATCH_PROCESSING_RESULTS[batch_id]["completed"] += 1
                print(f"✅ Successfully processed: {url}")
            else:
                BATCH_PROCESSING_RESULTS[batch_id]["failed"] += 1
                print(f"❌ Failed to process: {url}")
            
            # Add delay between requests to avoid rate limiting
            if i < len(instagram_urls) - 1:
                time.sleep(2)  # 2 second delay between requests
        
        # Mark batch as completed
        BATCH_PROCESSING_RESULTS[batch_id]["status"] = "completed"
        BATCH_PROCESSING_RESULTS[batch_id]["end_time"] = time.time()
        duration = BATCH_PROCESSING_RESULTS[batch_id]["end_time"] - BATCH_PROCESSING_RESULTS[batch_id]["start_time"]
        
        print(f"🎉 Batch {batch_id} completed in {duration:.2f} seconds")
        print(f"📊 Results: {BATCH_PROCESSING_RESULTS[batch_id]['completed']} successful, {BATCH_PROCESSING_RESULTS[batch_id]['failed']} failed")
        
    except Exception as e:
        print(f"💥 Fatal error in batch {batch_id}: {e}")
        BATCH_PROCESSING_RESULTS[batch_id]["status"] = "error"
        BATCH_PROCESSING_RESULTS[batch_id]["error"] = str(e)
        BATCH_PROCESSING_RESULTS[batch_id]["end_time"] = time.time()


# -----------------------------------------
# Flask API Endpoint Definition
# -----------------------------------------
                
@app.route("/download-video", methods=["POST"])
@require_api_key
def api_download_video():
    data = request.get_json()
    url = data.get("instagram_url")
    if not url:
        return jsonify({"error": "instagram_url required"}), 400
    path, folder = download_instagram_video(url)
    return jsonify({"video_path": path, "video_dir": folder, "logs": LOGS})

@app.route("/extract-frames", methods=["POST"])
@require_api_key
def api_extract_frames():
    data = request.get_json()
    path = data.get("video_path")
    if not path:
        return jsonify({"error": "video_path required"}), 400
    frames = extract_frames_from_video(path)
    return jsonify({"frames_count": len(frames), "frames": frames, "logs": LOGS})

@app.route("/transcribe-audio", methods=["POST"])
@require_api_key
def api_transcribe_audio():
    data = request.get_json()
    path = data.get("video_path")
    if not path:
        return jsonify({"error": "video_path required"}), 400
    transcript = transcribe_audio_assemblyai(ASSEMBLYAI_API_KEY, path)
    return jsonify({"transcript": transcript, "logs": LOGS})

@app.route("/process-gemini", methods=["POST"])
@require_api_key
def api_process_gemini():
    data = request.get_json()
    transcript = data.get("transcript", "")
    frames = data.get("frames", [])
    result = process_content_with_gemini(transcript, frames)
    return jsonify({"gemini_result": result, "logs": LOGS})

@app.route('/process-instagram-post', methods=['POST'])
@require_api_key
def api_process_instagram_post():
    start_request_time = time.time()
    log("--- Received request to /process-instagram-post ---")

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    instagram_url = data.get('instagram_url')
    if not instagram_url:
        return jsonify({"error": "'instagram_url' is required"}), 400

    try:
        result = process_instagram_post_logic(instagram_url)
        request_duration = time.time() - start_request_time
        log(f"--- Request to /process-instagram-post processed in {request_duration:.2f} seconds ---")

        status_code = 200 if result.get("success") else 500
        if result.get("error"):
             if "download" in result["error"].lower() or "URL" in result.get("details", "").lower():
                status_code = 422 # Unprocessable Entity
             elif "LLM" in result["error"] or "Gemini" in result.get("details", ""):
                status_code = 503 # Service unavailable (Gemini issue)
        return jsonify(result), status_code
    except Exception as e:
        log(f"FATAL: Unhandled exception in API endpoint: {e}")
        return jsonify({"error": "An internal server error occurred on the API.", "details": str(e)}), 500

@app.route('/process-instagram-posts-batch', methods=['POST'])
@require_api_key
def api_process_instagram_posts_batch():
    """NEW ENDPOINT: Accept array of URLs and process them in background"""
    start_request_time = time.time()
    print("--- Received request to /process-instagram-posts-batch ---")

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    instagram_urls = data.get('instagram_urls')
    
    if not instagram_urls or not isinstance(instagram_urls, list) or len(instagram_urls) == 0:
        return jsonify({"error": "'instagram_urls' must be a non-empty array of URLs"}), 400

    # Validate URLs
    valid_urls = []
    for url in instagram_urls:
        if isinstance(url, str) and ('instagram.com' in url and ('/p/' in url or '/reel/' in url or '/tv/' in url)):
            valid_urls.append(url)
    
    if not valid_urls:
        return jsonify({"error": "No valid Instagram URLs found in the provided array"}), 400
    
    # Generate unique batch ID
    batch_id = f"batch_{int(time.time())}_{os.urandom(4).hex()}"
    
    # Start background processing
    thread = threading.Thread(
        target=process_batch_in_background, 
        args=(batch_id, valid_urls),
        daemon=True
    )
    thread.start()
    
    request_duration = time.time() - start_request_time
    print(f"--- Batch {batch_id} queued for processing in {request_duration:.2f} seconds ---")
    
    # Return immediate response
    return jsonify({
        "success": True,
        "message": "Batch processing started successfully",
        "batch_id": batch_id,
        "total_urls": len(valid_urls),
        "valid_urls": len(valid_urls),
        "invalid_urls": len(instagram_urls) - len(valid_urls),
        "status_endpoint": f"/batch-status/{batch_id}"
    }), 202  # 202 Accepted - processing started

@app.route('/batch-status/<batch_id>', methods=['GET'])
@require_api_key
def api_batch_status(batch_id):
    """Get status of a batch processing job"""
    if batch_id not in BATCH_PROCESSING_RESULTS:
        return jsonify({"error": "Batch ID not found"}), 404
    
    batch_info = BATCH_PROCESSING_RESULTS[batch_id].copy()
    
    # Calculate progress percentage
    if batch_info["total"] > 0:
        batch_info["progress_percentage"] = round((batch_info["completed"] + batch_info["failed"]) / batch_info["total"] * 100, 2)
    else:
        batch_info["progress_percentage"] = 0
    
    # Calculate processing time
    if batch_info["status"] == "completed" and "end_time" in batch_info:
        batch_info["total_processing_time"] = round(batch_info["end_time"] - batch_info["start_time"], 2)
    else:
        batch_info["elapsed_time"] = round(time.time() - batch_info["start_time"], 2)
    
    return jsonify(batch_info), 200


# -----------------------------------------
# Main Execution Block (for local development)
# -----------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT')) # Adjusted port for local dev
    app.run(debug=True, host='0.0.0.0', port=port)