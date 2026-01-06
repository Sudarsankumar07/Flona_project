"""
CLI Script to run the complete pipeline from video_url.json
Run this directly: python run_pipeline.py
"""

import asyncio
import json
import time
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import BASE_DIR, OUTPUT_DIR, validate_config, get_config_summary
from ingestion import VideoDownloader
from transcription import Transcriber
from understanding import BRollCaptioner, EmbeddingGenerator
from matching import SemanticMatcher
from planning import TimelinePlanner
from schemas import BRollDescription


async def main():
    """Run the complete B-roll insertion pipeline"""
    
    print("=" * 60)
    print("  SMART B-ROLL INSERTER - PIPELINE")
    print("=" * 60)
    
    # Validate configuration
    try:
        validate_config()
        config = get_config_summary()
        print(f"\n✓ Configuration valid")
        print(f"  Vision Provider: {config.get('vision_provider')}")
        print(f"  Transcription: {config.get('transcription_provider', 'auto')}")
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("  Please check your .env file")
        return
    
    start_time = time.time()
    
    # =========================================================================
    # Step 1: Download videos from URLs
    # =========================================================================
    print("\n" + "-" * 60)
    print("[1/6] DOWNLOADING VIDEOS FROM URLs")
    print("-" * 60)
    
    downloader = VideoDownloader()
    json_path = BASE_DIR / "video_url.json"
    
    if not json_path.exists():
        print(f"✗ video_url.json not found at {json_path}")
        return
    
    download_results = await downloader.download_from_json(str(json_path))
    
    aroll_path = download_results["aroll"]["path"]
    aroll_duration = downloader.get_video_duration(aroll_path)
    aroll_filename = Path(aroll_path).name
    
    print(f"\n✓ A-roll: {aroll_filename} ({aroll_duration:.1f}s)")
    print(f"✓ B-rolls: {len(download_results['brolls'])} clips")
    
    # =========================================================================
    # Step 2: Transcribe A-roll
    # =========================================================================
    print("\n" + "-" * 60)
    print("[2/6] TRANSCRIBING A-ROLL")
    print("-" * 60)
    
    transcriber = Transcriber()
    
    # Check cache
    transcript_segments = transcriber.load_transcript(aroll_path)
    if transcript_segments:
        print(f"✓ Loaded cached transcript ({len(transcript_segments)} segments)")
    else:
        print("  Transcribing with AI...")
        transcript_segments = await transcriber.transcribe(aroll_path)
        print(f"✓ Transcribed: {len(transcript_segments)} segments")
    
    # Print transcript preview
    print("\n  Transcript preview:")
    for seg in transcript_segments[:3]:
        print(f"    [{seg.start:.1f}s-{seg.end:.1f}s] {seg.text[:60]}...")
    if len(transcript_segments) > 3:
        print(f"    ... and {len(transcript_segments) - 3} more segments")
    
    # =========================================================================
    # Step 3: Caption B-roll clips
    # =========================================================================
    print("\n" + "-" * 60)
    print("[3/6] CAPTIONING B-ROLL CLIPS")
    print("-" * 60)
    
    # Build broll_files list
    broll_files = []
    for broll in download_results["brolls"]:
        duration = downloader.get_video_duration(broll["path"])
        broll_files.append({
            "broll_id": broll["broll_id"],
            "filename": broll["filename"],
            "filepath": broll["path"],
            "duration": duration,
            "metadata": broll.get("metadata", "")
        })
    
    # Use metadata from JSON (pre-provided descriptions)
    broll_descriptions = []
    for broll in broll_files:
        metadata = broll.get("metadata", "").strip()
        if metadata:
            broll_descriptions.append(BRollDescription(
                broll_id=broll["broll_id"],
                filename=broll["filename"],
                description=metadata,
                duration=broll["duration"],
                filepath=broll["filepath"]
            ))
            print(f"  ✓ {broll['broll_id']}: Using provided metadata")
        else:
            # Would call AI here if no metadata
            captioner = BRollCaptioner()
            desc = await captioner.caption_single(
                filepath=broll["filepath"],
                broll_id=broll["broll_id"],
                filename=broll["filename"],
                duration=broll["duration"]
            )
            broll_descriptions.append(desc)
            print(f"  ✓ {broll['broll_id']}: AI generated caption")
    
    # =========================================================================
    # Step 4: Generate embeddings
    # =========================================================================
    print("\n" + "-" * 60)
    print("[4/6] GENERATING EMBEDDINGS")
    print("-" * 60)
    
    embedder = EmbeddingGenerator()
    
    print("  Embedding transcript segments...")
    transcript_embeddings = await embedder.embed_transcript_segments(transcript_segments)
    print(f"  ✓ {len(transcript_embeddings)} transcript embeddings")
    
    print("  Embedding B-roll descriptions...")
    broll_embeddings = await embedder.embed_broll_descriptions(broll_descriptions)
    print(f"  ✓ {len(broll_embeddings)} B-roll embeddings")
    
    # =========================================================================
    # Step 5: Semantic matching
    # =========================================================================
    print("\n" + "-" * 60)
    print("[5/6] FINDING SEMANTIC MATCHES")
    print("-" * 60)
    
    print("  Computing similarity matrix...")
    similarity_matrix = embedder.compute_similarity_matrix(
        transcript_embeddings,
        broll_embeddings
    )
    
    matcher = SemanticMatcher()
    selected_matches = matcher.find_best_matches(
        transcript_segments,
        broll_descriptions,
        similarity_matrix
    )
    
    print(f"  ✓ Found {len(selected_matches)} suitable insertion points")
    
    # Show matches
    for match in selected_matches:
        print(f"    • Segment {match.segment_id} → {match.best_broll_id} "
              f"(similarity: {match.similarity_score:.0%})")
    
    # =========================================================================
    # Step 6: Generate timeline
    # =========================================================================
    print("\n" + "-" * 60)
    print("[6/6] GENERATING TIMELINE")
    print("-" * 60)
    
    planner = TimelinePlanner()
    timeline = planner.generate_timeline(
        aroll_filename=aroll_filename,
        aroll_duration=aroll_duration,
        segments=transcript_segments,
        broll_descriptions=broll_descriptions,
        selected_matches=selected_matches,
        processing_start_time=start_time
    )
    
    # Validate
    validation = planner.validate_timeline(timeline)
    
    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    
    print(f"\n  A-roll Duration: {timeline.aroll_duration:.1f}s")
    print(f"  Total Insertions: {timeline.total_insertions}")
    print(f"  Processing Time: {timeline.processing_time_sec:.1f}s")
    
    print("\n  B-Roll Insertions:")
    print("  " + "-" * 56)
    
    for ins in timeline.insertions:
        print(f"  [{ins.start_sec:5.1f}s - {ins.start_sec + ins.duration_sec:5.1f}s] "
              f"{ins.broll_filename:<15} (confidence: {ins.confidence:.0%})")
        print(f"      Context: \"{ins.transcript_text[:50]}...\"")
    
    print("\n  " + "-" * 56)
    print(f"  Timeline saved to: {OUTPUT_DIR / 'timeline_latest.json'}")
    
    if validation["warnings"]:
        print(f"\n  ⚠ Warnings: {len(validation['warnings'])}")
        for warn in validation["warnings"]:
            print(f"    - {warn}")
    
    print("\n" + "=" * 60)
    
    return timeline


if __name__ == "__main__":
    asyncio.run(main())
