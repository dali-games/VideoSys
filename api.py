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
        try:
            config = CogVideoXConfig(request.model_name, enable_pab=True)
            engine = VideoSysEngine(config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing the model: {str(e)}")

        # Generate the video based on the provided parameters
        result = engine.generate(
            prompt=request.prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            num_frames=request.num_frames,
            seed=request.seed,
        )

        # Check if the result is valid
        if result is None or not hasattr(result, 'video') or not result.video:
            raise HTTPException(status_code=500, detail="Video generation returned no data.")

        video = result.video[0] if result.video else None

        if video is None:
            raise HTTPException(status_code=500, detail="No video data received.")

        # Save the video to a unique file
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)

        unique_filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(output_dir, unique_filename)

        if output_path is None:
            raise HTTPException(status_code=500, detail="Failed to construct the output path.")

        engine.save_video(video, output_path)

        if not os.path.isfile(output_path):
            raise HTTPException(status_code=500, detail="Failed to save the video file.")

        # Return the video file as a response
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=unique_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
