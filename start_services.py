#!/usr/bin/env python3
"""Start Rapid Detector web application."""

import uvicorn

def main():
    """Entry point for rapid-detector command."""
    print("🚀 Starting Rapid Detector...")
    print("🔗 Web UI: http://localhost:8000/")
    print("⚡ API: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")
    
    try:
        from app.api import app
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except KeyboardInterrupt:
        print("\n👋 Stopping service...")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
