# US Drug Overdose Rate Map

Interactive county-level visualization of US drug overdose rates comparing two 5-year ACS estimates (2018–2022 vs 2020–2024).

## Features
- **National view** → click any state to zoom in
- **State view** → click any county for detailed stats
- **Sidebar panel** with overdose rates, change, and socioeconomic variables
- **Mode toggle** between 2018–2022, 2020–2024, and Change

---

## Setup Instructions

### Step 1 — Run the Colab script
1. Open `overdose_colab.py` in Google Colab
2. Upload your two CSV files when prompted
3. Download the output `counties.json`

### Step 2 — Set up folder
```
overdose-map/
├── index.html          ← already done
├── README.md           ← this file
└── data/
    └── counties.json   ← paste your downloaded file here
```

### Step 3 — Push to GitHub
```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/overdose-map.git
git push -u origin main
```

### Step 4 — Enable GitHub Pages
1. Go to your repo on GitHub
2. Settings → Pages
3. Source: **Deploy from a branch**
4. Branch: **main** / root
5. Save → wait ~1 min → your site is live at:
   `https://YOUR_USERNAME.github.io/overdose-map`

---

## Data Notes
- Overdose data: CDC WONDER (suppressed for counties with <10 deaths)
- Socioeconomic data: American Community Survey 5-Year Estimates
- "5-year estimate" means values represent a rolling average, not a single year
