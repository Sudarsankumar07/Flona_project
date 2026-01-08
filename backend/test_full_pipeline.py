"""Test the full pipeline from KeywordMatcher to TimelinePlanner"""
import sys
sys.path.insert(0, '.')

from schemas import TranscriptSegment, BRollDescription
from matching import KeywordMatcher
from planning import TimelinePlanner

# Test data
segments = [
    TranscriptSegment(id=1, start=0.0, end=4.1, text='Hi everyone! Today, we are gonna talk about food quality and safety.'),
    TranscriptSegment(id=2, start=4.1, end=8.3, text='It is super important to know what you are eating, right? Like, is it healthy?'),
    TranscriptSegment(id=3, start=8.3, end=12.1, text='So, first things first, always check the expiry date on packaged foods.'),
    TranscriptSegment(id=4, start=12.1, end=15.8, text='Expiry date is like, the last date you should eat that thing.'),
    TranscriptSegment(id=5, start=15.8, end=20.2, text='And, you know, look for any signs of spoilage. Like, if it smells weird, do not eat it!'),
    TranscriptSegment(id=6, start=20.2, end=24.5, text='Next up, hygiene! Before cooking or eating, wash your hands properly.'),
    TranscriptSegment(id=7, start=24.5, end=28.3, text='Use soap and water, and scrub for at least 20 seconds. It is a must!'),
    TranscriptSegment(id=8, start=28.3, end=32.4, text='Also, when you are buying fruits and veggies, try to choose fresh and seasonal ones.'),
    TranscriptSegment(id=9, start=32.4, end=36.5, text='They are usually more nutritious and less likely to have harmful chemicals.'),
    TranscriptSegment(id=10, start=36.5, end=40.1, text='So yeah, just be mindful of these things, and stay healthy!'),
]

brolls = [
    BRollDescription(broll_id='broll_1.mp4', filename='broll_1.mp4', description='a street scene with a food stand', duration=4.0, filepath='broll_1.mp4'),
    BRollDescription(broll_id='broll_2.mp4', filename='broll_2.mp4', description='a plastic container on a table next to a paper bag', duration=4.0, filepath='broll_2.mp4'),
    BRollDescription(broll_id='broll_3.mp4', filename='broll_3.mp4', description='food being served in plastic containers on a table', duration=4.0, filepath='broll_3.mp4'),
    BRollDescription(broll_id='broll_4.mp4', filename='broll_4.mp4', description='two plates of food on a kitchen counter', duration=4.0, filepath='broll_4.mp4'),
    BRollDescription(broll_id='broll_5.mp4', filename='broll_5.mp4', description='a table filled with pastries', duration=4.0, filepath='broll_5.mp4'),
    BRollDescription(broll_id='broll_6.mp4', filename='broll_6.mp4', description='a bowl of fruit and a glass of water on a table', duration=4.0, filepath='broll_6.mp4'),
]

# Run keyword matcher
print("Running KeywordMatcher...")
matcher = KeywordMatcher()
insertions = matcher.find_matches(segments, brolls, max_insertions=6, min_gap_seconds=3.0)

print(f'KeywordMatcher returned {len(insertions)} insertions')
if insertions:
    print(f'Type: {type(insertions[0]).__name__}')

# Create timeline
print("\nCreating timeline...")
planner = TimelinePlanner()
timeline = planner.create_timeline(
    aroll_filename='a_roll.mp4',
    aroll_duration=40.12,
    matches=insertions,
    transcript_segments=segments,
    broll_descriptions=brolls
)

print(f'Timeline created with {timeline.total_insertions} insertions')

# Save it
print("\nSaving timeline...")
planner.save_timeline(timeline)
print('Done!')

# Show insertions
print(f"\n=== {timeline.total_insertions} Insertions ===")
for ins in timeline.insertions:
    print(f"  {ins.start_sec}s: {ins.broll_id} ({ins.duration_sec}s) - {ins.reason[:40]}...")
