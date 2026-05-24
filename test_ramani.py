import os
from ousia.session_processor import SessionProcessor
from ousia.visualizer import quick_plot
from dotenv import load_dotenv

load_dotenv()

def run_test():
    print("🚀 Ousia Dr. Ramani Session Analysis Started (First 5 Minutes)")
    
    # Initialize processor
    # We use 'base' whisper for speed. grok-3 for LLM.
    # subject_speaker is typically SPEAKER_01 if the intro is by the host (Ramani)
    # but let's see what the diarizer finds.
    sp = SessionProcessor(
        whisper_model="base",
        llm_model="grok-3",
        hf_token=os.getenv("HF_TOKEN"),
        subject_speaker="SPEAKER_01"  # Guessing subject is speaker 01
    )
    
    audio_path = "samples/ramani_5min.wav"
    
    if not os.path.exists(audio_path):
        print(f"❌ Error: {audio_path} not found.")
        return

    print(f"📦 Processing {audio_path}...")
    graph = sp.process(audio_path)
    
    print("\n📊 Graph Summary:")
    graph.summary()
    
    # Save the characteristic graph
    output_path = "output/ramani_characteristic_graph.png"
    print(f"🎨 Visualizing to {output_path}...")
    quick_plot(graph, save_path=output_path)
    print("✅ Analysis Complete.")

if __name__ == "__main__":
    run_test()
