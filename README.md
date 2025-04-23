=======
# Bowliverse2.0

A biomechanics analysis platform for cricket bowling (fast, spin, etc.), evaluating legality and performance under diverse conditions.

## Setup
1. Clone: `git clone https://github.com/rniyer0510/bowliverse2.0.git`
2. Create virtual environment: `python -m venv env`
3. Activate: `source env/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Run Analysis
```bash
python -m scripts.analyze_video videos/fast_61ukNhaghRo.mp4 videos output output/hmm_release_elbow_fast.pkl fast
>>>>>>> febdb22 (Initial commit of Bowliverse2.0 codebase)
