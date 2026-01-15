from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "API is alive!"}

@app.get("/status")
def check_status():
    return {"status": "The sensors are pulsing."}