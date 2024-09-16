from videosys import CogVideoXConfig, VideoSysEngine

def run_pab():
    config = CogVideoXConfig("THUDM/CogVideoX-5b", enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(
        prompt=prompt,
        guidance_scale=6,
        num_inference_steps=50,
        num_frames=49,
        seed=42,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_pab()
