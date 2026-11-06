# Undertone

**Different song. Same feeling.** A complete music discovery website with a reproducible audio-feature recommendation pipeline and 81,343 recordings.

Search by song or artist, choose a recording, and explore similar tracks. Fine-tune BPM, mood, vocal type, genre, acousticness, and explicit content. Save discoveries locally and open tracks on Spotify. No API keys, server, or visitor account required.

## Run locally

Install Node.js 22+ and run `npm run dev`. Open http://127.0.0.1:4173. The committed `dist/` contains the full deployable application and trained metric, so no training or npm dependencies are needed just to serve it.

`npm test` checks the real catalogue, search, exclusion rules, vocal constraints, and mood/BPM filters. For browser checks on Windows with Edge installed, run `npm ci` then `node tests/browser.mjs`. Screenshots are written to ignored `artifacts/`. `TEST_URL` can point browser tests at a deployed URL.

## Recommendation model

The pipeline fits a standard scaler and selects a weighted nearest-neighbor metric from four candidates on an artist-separated validation set. Nine features represent tempo, energy, valence, danceability, acousticness, instrumentalness, speechiness, loudness, and liveness. Ranking uses standardized weighted Euclidean distance; hard filters apply before ranking. Duplicate track IDs and artist/title reissues are collapsed. Results are capped at two tracks per artist.

Evaluation partitions by the primary artist's SHA256 bucket (60% reference, 20% validation, 20% test). Each metric uses the same 1,200 validation queries. Test queries are held out until selection. The scaler is refit on the complete catalogue for deployment. See [the machine-readable model report](dist/model-report.json) for actual results, source hash, selected weights, and limitations.

**This is an evaluated content-based baseline, not a claim of state-of-the-art music understanding.** Genre overlap@10 is a weak proxy for perceptual similarity. There are no listener relevance judgments, raw audio embeddings, collaborative listening signals, or trained emotion classifiers. Valence and energy provide heuristic mood labels. A stronger model requires properly licensed audio/embeddings, listener-labeled similar-song pairs, and listening evaluations, with sufficient training and inference resources.

### Reproduce training

Install Python 3.11+ and `pip install -r requirements.txt`, then `python scripts/train.py`. It downloads a pinned dataset revision if `data/tracks.csv` is missing, cleans the catalogue, evaluates candidates, and writes `dist/catalog.json` and `dist/model-report.json`. Runtime metadata is compacted into array rows; the feature offsets are documented in `dist/engine.js`. The training seed is 42. Exact numerical library versions are recorded in the report. The raw download is ignored by Git.

### Vocal filters and catalogue coverage

The source has **no vocal-type ground truth**. `data/vocals.json` supplies a limited, editable list of editorial artist-level lead-vocal hints. These are approximate and may be wrong on individual tracks. Unlisted artists and collaborations remain unknown, and unknowns are excluded from male/female/mixed filters. Tracks with instrumentalness ≥ 0.9 are labeled “likely instrumental.” There is no voice gender inference. Broader accurate filtering needs song-level annotations.

The historical catalogue has 81,343 unique recordings after cleaning; it cannot find every song or recent release. BPM can be ambiguous at half/double time. Recommendations do not include hosted audio. Spotify playback and regional availability are controlled by Spotify.

## Architecture and privacy

Buildless HTML, CSS, and ES modules; a Web Worker loads the catalogue and performs search/ranking without blocking the interface. All search terms and saved tracks remain in the visitor's browser. No analytics or advertising. Page requests go to the hosting provider, fonts to Google Fonts, and Listen links to Spotify. Saved collections use local storage and can be removed by clearing site data.

`dist/` is portable to any static host, including GitHub Pages. All asset paths are relative to support repository subpaths. `.github/workflows/pages.yml` tests and publishes the site on pushes to `main` after GitHub Pages is configured to use GitHub Actions. `.openai/hosting.json` binds the corresponding Sites deployment. The website requires no paid API or application backend.

## Data and credits

Source: [Maharshi Pandya's Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset), revision `635b034f69257814eff850a5c2b3346fe458134f`. Its dataset card declares a BSD license and describes collection through Spotify's Web API. The original card is retained in `data/DATASET_CARD.md`; the report records the downloaded CSV SHA256. Track names and artist names belong to their respective owners; no audio, lyrics, or album artwork is redistributed. Generated cover tiles are abstract artwork, not album covers.

Application source is licensed under MIT; source data remains subject to its upstream terms. Review upstream permissions before commercial reuse or expanding training data.
