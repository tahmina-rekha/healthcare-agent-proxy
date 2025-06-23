# main.py (for your 'agent-proxy-service' Cloud Run service)
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS

# Ensure google-auth and requests are installed via requirements.txt
from google.auth import default
from google.auth.transport.requests import Request as GoogleAuthRequest

app = Flask(__name__)
CORS(app) # Enable CORS for your Flask app, allowing requests from any origin

# NEW: Add a simple root route to handle OPTIONS/GET requests to the base URL
# This is crucial for CORS preflight requests that might hit the root.
@app.route('/', methods=['GET', 'OPTIONS'])
def hello_world():
    """
    A simple root endpoint to confirm the service is running and to handle
    CORS preflight (OPTIONS) requests to the base URL.
    """
    if request.method == 'OPTIONS':
        # CORS preflight will be handled by flask_cors automatically due to CORS(app)
        return '', 204 # No Content
    return jsonify({"message": "Agent proxy service is running!"}), 200

# This is the primary endpoint your frontend will call
@app.route('/chat_with_agent', methods=['POST'])
def chat_with_agent():
    """
    Receives user messages from the frontend, forwards them to the Vertex AI Agent Builder agent,
    and returns the agent's response. Handles authentication to the Dialogflow API.
    """
    try:
        request_data = request.get_json()

        # Extract necessary information from the frontend request
        user_message = request_data.get('message')
        session_id = request_data.get('sessionId', 'default-session') # Use a dynamic session ID in production
        project_id = request_data.get('projectId')
        location = request_data.get('location')
        agent_id = request_data.get('agentId')

        # Basic validation
        if not all([user_message, project_id, location, agent_id]):
            return jsonify({"error": "Missing required parameters (message, projectId, location, agentId)."}), 400

        # Construct the API endpoint URL for your agent's predict method
        # This endpoint is specific to Dialogflow CX (which Agent Builder uses underneath)
        dialogflow_api_url = f"https://{location}-dialogflow.googleapis.com/v3/projects/{project_id}/locations/{location}/agents/{agent_id}:predict"

        # Get Google Cloud credentials automatically from the environment (Cloud Run handles this)
        # These credentials are for the Cloud Run service account, allowing it to call Dialogflow.
        from google.auth import default
        from google.auth.transport.requests import Request as GoogleAuthRequest # Renamed for clarity if 'requests' is also imported
        credentials, project = default()
        auth_req = GoogleAuthRequest()
        credentials.refresh(auth_req) # Ensure credentials are fresh

        # Prepare the payload for the Dialogflow API
        dialogflow_payload = {
            "queryInput": {
                "text": {
                    "text": user_message, # Corrected: 'text' key is now properly quoted
                    "languageCode": 'en-US', # Corrected: 'languageCode' key is now properly quoted
                },
            },
            "sessionId": session_id,
        }

        # Make the request to the Dialogflow API
        import requests
        headers = {
            'Authorization': f'Bearer {credentials.token}',
            'Content-Type': 'application/json'
        }
        
        dialogflow_response = requests.post(dialogflow_api_url, headers=headers, json=dialogflow_payload)
        dialogflow_response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        dialogflow_data = dialogflow_response.json()

        # Extract the agent's response text from the Dialogflow response
        agent_response_text = dialogflow_data.get('queryResult', {}).get('responseMessages', [{}])[0].get('text', {}).get('text', [''])[0] or \
                              dialogflow_data.get('queryResult', {}).get('fulfillmentText', '') or \
                              "I'm sorry, I couldn't get a clear response from the agent via the proxy."

        return jsonify({"response": agent_response_text}), 200

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error from Dialogflow API: {e.response.status_code} - {e.response.text}")
        return jsonify({"error": f"Error from agent API: {e.response.status_code} - {e.response.text}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred in the proxy: {e}")
        return jsonify({"error": f"Proxy error: {str(e)}"}), 500

# Standard boilerplate for running a Flask application.
# Cloud Run automatically sets the 'PORT' environment variable.
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
