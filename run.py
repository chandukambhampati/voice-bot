import uvicorn

if __name__ == "__main__":
    print("Starting Advanced Cancer Center - AI Voice Bot Dashboard...")
    print("Open http://127.0.0.1:8000 in your browser to interact with the bot.")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
