# Keyword Planner: SEO Superhero Unleashed! 🦸‍♂️

![Keyword Planner Banner](https://raw.githubusercontent.com/FreeMarketamilitia/Keyword-Planner/main/images/banner.png)
*Caption: "Because who needs a cape when you’ve got keywords?"*

![PyPI](https://img.shields.io/pypi/v/keyword-planner?color=blue) 
![License](https://img.shields.io/github/license/FreeMarketamilitia/Keyword-Planner) 
![Sass Level](https://img.shields.io/badge/Sass-Level%20100-purple) 
![SEO Power](https://img.shields.io/badge/SEO%20Power-Over%209000-red) 
![Stars](https://img.shields.io/github/stars/FreeMarketamilitia/Keyword-Planner?style=social) 
![Forks](https://img.shields.io/github/forks/FreeMarketamilitia/Keyword-Planner?style=social) 
![AI Swagger](https://img.shields.io/badge/AI%20Swagger-Gemini%20Approved-green) 
![Bug Slayer](https://img.shields.io/badge/Bugs-Slayed%20Daily-orange) 
![Code Quality](https://img.shields.io/badge/Code%20Quality-Snaccidentally%20Good-blueviolet) 
![Coffee Fuel](https://img.shields.io/badge/Fueled%20By-Coffee%20&%20Chaos-brown)

Welcome to the **Keyword Planner**, a Flask-powered beast that’s here to kick your SEO game into overdrive! Think of it as your snarky sidekick, blending **Google Search Console**, **Google Ads Keyword Planner**, and **Gemini AI** to sniff out keyword gold and roast your competitors’ weak spots. Buckle up—this ain’t your grandma’s keyword tool! 😏

---

## Table of Contents (Because We’re Fancy Like That)

- [Features](#features) 🎉
- [Installation](#installation) 🛠️
- [Usage](#usage) 🚀
- [Configuration](#configuration) 🔧
- [Screenshots](#screenshots) 📸
- [Contributing](#contributing) 🤝
- [License](#license) 📜

---

## Features: Why We’re the Cool Kids 😎

| **Feature**              | **Why It’s Awesome**                                              |
|--------------------------|-------------------------------------------------------------------|
| **Keyword Extraction**   | Snags juicy data straight from Google Search Console. Yum! 🍔     |
| **Competitor Analysis**  | Spies on rivals like a ninja 🥷, finding gaps they didn’t plug.   |
| **Keyword Variations**   | Gemini AI + NLTK = keyword combos so hot they sizzle! 🔥          |
| **Social Validation**    | Fakes X vibes ‘til we get the real deal. Stay tuned, fam! 📱     |
| **Clustering & Scoring** | KMeans magic + custom scores = keyword royalty crowned. 👑       |
| **Web Interface**        | Flask UI so slick, it’s basically a fashion show runway. 💃      |
| **Export Options**       | CSV & JSON exports—because data hoarders deserve love too. 💾     |

*“SEO’s a jungle, and we’re your machete-wielding guide!”*

---

## Installation: Let’s Get This Party Started 🎉

### Prerequisites (The VIP List)

- **Python**: 3.8+ (because we’re not stuck in 2010, duh)
- **Git**: For snagging this beauty from the interwebs
- **OS**: Ubuntu or any Linux flavor—Windows users, we’ll pray for you 🙏

### Steps (No PhD Required)

#### tl:dr

```bash 
pip install Keyword-Kraken
 ```

1. **Clone the Repository**
- ```bash 
   git clone https://github.com/FreeMarketamilitia/Keyword-Planner.git && cd Keyword-Planner
    ```
   - *Like stealing candy from a GitHub repo!*

2. **Set Up Virtual Environment**
- ```bash
   python3 -m venv venv && source venv/bin/activate
  ```  
   - *Because we don’t mix our potions with the common folk.*

3. **Install Dependencies**
- ```bash
    pip install -r requirements.txt
   ```
   ```
     flask           # Web magic, baby
     pandas          # Data wrangling pro
     google-auth-oauthlib  # Google’s bouncer
     google-api-python-client  # Google’s VIP pass
     google-ads      # Ad keyword ninja
     google-generativeai  # AI with sass
     python-dotenv   # Keeps secrets safe
     tenacity        # Retry like a boss
     nltk            # Word nerd toolkit
     scikit-learn    # Math geek vibes
     requests        # Web stalker tool
     beautifulsoup4  # HTML soup chef
     ```

4. **Download NLTK Data**
- ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```
- *Because words need love too!*

5. **Configure API Keys**
   - Check [API_SETUP_README.md](API_SETUP_README.md) to bribe Google and Gemini with keys. 🗝️

---

## Usage: How to Slay Like a Pro 🗡️

1. **Run the Application**
   - ```python
      python app.py
     ``` 
   - *Hit it like it’s hot!*

2. **Access the Web Interface**
   - Surf to `http://localhost:5000` like a boss. 🌐

3. **Input Parameters**
   - Toss in:
     - **Site URL**: Your digital turf
     - **Date Range**: When to snoop
     - **Settings**: Iterations, weights—tweak it ‘til it sings! 🎶
   - Smash that *Submit* button like it owes you money.

4. **View Results**
   - Scroll a sexy paginated table.
   - Grab **CSV** or **JSON**—because who doesn’t love a data snack? 🍕

*“You’re basically Tony Stark with keywords now.”*

---

## Configuration: Tweak It, Geek It! 🔧

Edit `.env` per [API_SETUP.md](API_SETUP_README.md). Here’s the VIP table:

| **Variable**              | **What It Does**                          |
|---------------------------|-------------------------------------------|
| `GEMINI_API_KEY`          | Unlocks Gemini AI’s snarky genius         |
| `GOOGLE_ADS_YAML_PATH`    | Points to Google Ads’ secret stash        |
| `SEARCH_CONSOLE_JSON_PATH`| Opens Search Console’s treasure chest     |
| `SECRET_KEY`              | Flask’s secret handshake (optional magic) |

- **Pro Tip**: Edit `competitor_urls.json` to stalk your rivals like a pro spy. 🕵️‍♂️

