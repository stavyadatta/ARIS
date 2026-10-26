import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

IMG_DIR = os.getenv("DISPLAY_IMGS_DIR", "/workspace/display_imgs")
os.makedirs(IMG_DIR, exist_ok=True)
SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the images directly
app.mount("/display_imgs", StaticFiles(directory=IMG_DIR), name="display_imgs")

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    # Find the latest image in the directory (or just the first)
    image_files = [f for f in sorted(os.listdir(IMG_DIR)) if f.lower().endswith(SUPPORTED_EXTS)]
    if not image_files:
        html = "<h2>No images found in display_imgs/</h2>"
    else:
        # Always pick the newest or the first
        latest = image_files[-1]
        html = f"""
        <html>
        <head>
            <title>Live Image Viewer</title>
            <meta http-equiv="refresh" content="2"> <!-- Refresh every 2s -->
        </head>
        <body>
            <h2>Currently displaying: <code>{latest}</code></h2>
            <img src="/display_imgs/{latest}" style="max-width:90vw;max-height:80vh;"/>
            <p>Put a new image in <code>display_imgs/</code> to update display.<br>
            The browser will auto-refresh every 2 seconds.</p>
        </body>
        </html>
        """
    return HTMLResponse(content=html)

# Optional: API to list files (if you want to extend later)
@app.get("/list_images")
async def list_images():
    files = [f for f in sorted(os.listdir(IMG_DIR)) if f.lower().endswith(SUPPORTED_EXTS)]
    return {"images": files}

def image_serve():
    import uvicorn
    uvicorn.run("image_viewer:app", host="0.0.0.0", port=8003, reload=False, log_level="critical")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("image_viewer:app", host="0.0.0.0", port=8003, reload=True, log_level="critical")
