# AegisRoute AI - FastAPI Backend

This is the Python FastAPI backend for AegisRoute AI. It handles hazard polygon state, shelter capacities, CAP broadcast alerts, and AI obstacle-avoidance evacuation route generation.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the FastAPI server on port 8000:
```bash
uvicorn main:app --reload --port 8000
```

3. Interactive Swagger API Docs:
Open `http://localhost:8000/docs` in your browser.

4. Connect Frontend:
In the root `.env` file of the React application, set:
```env
VITE_API_BASE_URL=http://localhost:8000
```
The React app will automatically connect to this FastAPI backend!
