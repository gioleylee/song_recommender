"""Reproducible audio-feature metric selection. No listening-quality claims."""
import json, hashlib, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/635b034f69257814eff850a5c2b3346fe458134f/dataset.csv'
FEATURES = ['tempo','energy','valence','danceability','acousticness','instrumentalness','speechiness','loudness','liveness']

def main():
    raw = ROOT / 'data/tracks.csv'
    raw.parent.mkdir(exist_ok=True)
    if not raw.exists(): urllib.request.urlretrieve(SOURCE, raw)
    df = pd.read_csv(raw).dropna(subset=['track_id','track_name','artists'] + FEATURES)
    # Collapse reissues and repeated genre rows, keeping genre memberships.
    genres = df.groupby('track_id').track_genre.agg(lambda x: sorted(set(x))).to_dict()
    df = df.sort_values('popularity', ascending=False).drop_duplicates('track_id')
    df = df.drop_duplicates(['artists','track_name']).reset_index(drop=True)
    x = df[FEATURES].to_numpy(float)
    # Artist groups are assigned deterministically, keeping all collaborations together by primary artist.
    groups = df.artists.str.split(';').str[0]
    buckets = groups.map(lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%10).to_numpy()
    fit, valid, test = buckets < 6, (buckets >= 6)&(buckets < 8), buckets >= 8
    scaler = StandardScaler().fit(x[fit])
    z = scaler.transform(x)
    labels = [set(genres[t]) for t in df.track_id]
    candidates = {
      'balanced': [1,1,1,1,1,1,1,1,1],
      'mood-first': [1.2,2.0,2.5,0.8,1.4,1.0,0.5,0.5,0.3],
      'texture-first': [0.6,1,1,1,2,2,1,0.5,0.5],
      'rhythm-first': [2,1.5,1,2,1,1,0.7,0.5,0.3],
    }
    ref = np.flatnonzero(fit)
    def evaluate(mask, weights):
        rng = np.random.default_rng(42)
        query = np.flatnonzero(mask)
        query = rng.choice(query, min(1200,len(query)), replace=False)
        w = np.sqrt(weights)
        nn = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(z[ref]*w)
        near = nn.kneighbors(z[query]*w, return_distance=False)
        return float(np.mean([len([j for j in row if labels[ref[j]] & labels[q]])/10 for q,row in zip(query,near)]))
    scores = {k:evaluate(valid,w) for k,w in candidates.items()}
    winner = max(scores, key=scores.get)
    test_score = evaluate(test, candidates[winner])
    # Final scaler refit on the complete deployment catalogue after selection.
    scaler.fit(x)
    out = ROOT/'dist'; out.mkdir(exist_ok=True)
    voice = json.loads((ROOT/'data/vocals.json').read_text(encoding='utf-8'))
    rows = []
    for r in df.to_dict('records'):
        artists = r['artists'].split(';')
        v = 'instrumental' if r['instrumentalness'] >= .9 else 'unknown'
        if v != 'instrumental' and len(artists) == 1: v = voice.get(artists[0], 'unknown')
        rows.append([r['track_id'], r['track_name'],r['artists'],r['album_name'],int(r['popularity']),bool(r['explicit']),genres[r['track_id']],v]+[round(float(r[f]),4) for f in FEATURES])
    model = dict(features=FEATURES, mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),weights=candidates[winner],name=winner)
    report = dict(source=SOURCE,source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),tracks=len(rows),validation_precision_at_10=scores,selected=winner,test_genre_overlap_at_10=test_score,split='Primary artist SHA256 bucket: 60% reference, 20% validation, 20% test',evaluation_queries=1200,seed=42,limitations=['Genre overlap is a proxy, not listener preference or mood ground truth.','Historical catalogue; not a live Spotify integration.','Vocal labels are editorial artist-level hints, not song-level predictions. Unknown collaborations are excluded from vocal filters.','Valence and energy are proxies for mood.','No raw audio or listening-history training.'])
    report['library_versions'] = {'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__}
    (out/'catalog.json').write_text(json.dumps(dict(model=model,tracks=rows),separators=(',',':'),ensure_ascii=False),encoding='utf-8')
    (out/'model-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
