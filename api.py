from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from videosys import CogVideoXConfig, VideoSysEngine
from fastapi.responses import FileResponse
import os
import uuid

app = FastAPI(debug=True)

class VideoRequest(BaseModel):
    model_name: str = "THUDM/CogVideoX-5b"  # Default model name
    prompt: str
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    num_frames: int = 49
    seed: int = 42

@app.post("/generate_video")
def generate_video(request: VideoRequest):
    try:
        # Initialize configuration and engine with the specified model name
        config = CogVideoXConfig(request.model_name, enable_pab=True)
        engine = VideoSysEngine(config)

        # Generate the video based on the provided parameters
        video = engine.generate(
            prompt=request.prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            num_frames=request.num_frames,
            seed=request.seed,
        ).video[0]

        # Save the video to a unique file
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        unique_filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(output_dir, unique_filename)
        engine.save_video(video, output_path)

        # Return the video file as a response
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=unique_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
