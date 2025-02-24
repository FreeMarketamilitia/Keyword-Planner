from flask import Flask, render_template, request, send_from_directory, abort, session
import pandas as pd
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import time
import json
from google_auth_oauthlib.flow import InstalledAppFlow
import requests
from bs4 import BeautifulSoup
import re
import random
import urllib.robotparser
from google.auth.transport.requests import Request


def configure_logging(enable_debug=False):
    """Configure logging with specified level and format."""
    level = logging.DEBUG if enable_debug else logging.INFO
    logging.basicConfig(
        filename="keyword_planner.log",
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger(__name__)

load_dotenv()
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_ADS_YAML_PATH = os.getenv("GOOGLE_ADS_YAML_PATH")
SEARCH_CONSOLE_JSON_PATH = os.getenv("SEARCH_CONSOLE_JSON_PATH")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env")
if not GOOGLE_ADS_YAML_PATH or not os.path.exists(GOOGLE_ADS_YAML_PATH):
    raise ValueError("GOOGLE_ADS_YAML_PATH not set or file not found in .env")
if not SEARCH_CONSOLE_JSON_PATH or not os.path.exists(SEARCH_CONSOLE_JSON_PATH):
    raise ValueError("SEARCH_CONSOLE_JSON_PATH not set or file not found in .env")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(RESULTS_FOLDER).mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SESSION_COOKIE_SECURE"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


def fetch_page_content(url, respect_robots_txt=True):
    """Fetch page content while respecting robots.txt if specified."""
    # TODO: Add caching mechanism to avoid repeated requests for the same URL
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        if respect_robots_txt:
            rp = urllib.robotparser.RobotFileParser()
            base_url = f"{url.split('/')[0]}//{url.split('/')[2]}"
            rp.set_url(f"{base_url}/robots.txt")
            rp.read()
            if not rp.can_fetch(headers["User-Agent"], url):
                logger.warning(f"Blocked by robots.txt: {url}")
                return ""

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(
            [elem.get_text(strip=True) for elem in soup.find_all(["p", "h1", "h2", "h3"])]
        )
        return text[:2000]
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {str(e)}")
        return ""


def get_trends_score(keywords, initial_data):
    """Estimate trends score based on historical impressions."""
    # TODO: Incorporate time-series analysis for more accurate trend prediction
    logger.debug(f"Estimating trends for keywords: {keywords[:5]}...")
    trends_scores = {}
    historical_data = initial_data.groupby("keyword").agg(
        {"impressions": "sum", "clicks": "sum", "ctr": "mean"}
    ).reset_index()

    for kw in keywords:
        if kw in historical_data["keyword"].values:
            kw_data = historical_data[historical_data["keyword"] == kw]
            impressions = kw_data["impressions"].iloc[0]
            trends_scores[kw] = min(
                100, impressions / max(historical_data["impressions"].max(), 1) * 100
            )
        else:
            trends_scores[kw] = 0
    logger.info(f"Estimated trends for {len(trends_scores)} keywords")
    return trends_scores


def load_competitor_urls():
    """Load competitor URL templates from JSON file."""
    # TODO: Add error handling for missing or malformed JSON file
    with open("competitor_urls.json", "r") as file:
        return json.load(file)


def crawl_competitor_content(keyword, exclude_url, num_results, respect_robots_txt=True):
    """Crawl competitor content for a given keyword."""
    # TODO: Parallelize crawling to improve performance for large num_results
    competitor_templates = load_competitor_urls()
    competitor_urls = [
        template.format(keyword=keyword.replace(" ", "+"))
        for template in competitor_templates
    ]

    content_dict = {}
    fetched = 0

    for url in competitor_urls:
        if exclude_url in url:
            continue
        content = fetch_page_content(url, respect_robots_txt)
        if content:
            content_dict[url] = content
            fetched += 1
        if fetched >= num_results:
            break
        time.sleep(1)

    return content_dict


@retry(
    stop_after_attempt(3),
    wait_exponential(multiplier=1, min=4, max=60),
    retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def analyze_competitor_gaps(base_keywords, site_url, num_results, respect_robots_txt=True):
    """Analyze gaps in competitor content for given keywords."""
    # TODO: Enhance gap analysis with NLP to detect specific topics or entities
    logger.debug(f"Analyzing competitor gaps for {base_keywords[:5]} with {num_results} results...")
    gaps = {}
    for kw in base_keywords:
        logger.info(f"Crawling for keyword: {kw}")
        competitor_content_dict = crawl_competitor_content(
            kw, site_url, num_results, respect_robots_txt
        )

        if not competitor_content_dict:
            gaps[kw] = "No competitor content found"
            continue

        combined_content = " ".join(competitor_content_dict.values())
        if combined_content:
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = (
                f"Analyze this competitor content for '{kw}': '{combined_content[:2000]}'. "
                "Identify untapped opportunities (e.g., unanswered questions, missing topics) in 1-2 sentences."
            )
            try:
                response = model.generate_content(prompt)
                gaps[kw] = response.text.strip() or "No clear gaps identified"
            except Exception as e:
                logger.error(f"Gemini analysis error: {str(e)}")
                gaps[kw] = "Analysis failed"
        else:
            gaps[kw] = "No content fetched from competitors"

        logger.info(f"Gap for '{kw}': {gaps[kw]}")
        time.sleep(1)

    return gaps


def synonym_expansion(base_keywords, num_variations):
    """Generate keyword variations using synonyms."""
    # TODO: Filter synonyms by relevance or context using word embeddings
    variations = []
    for kw in base_keywords:
        words = kw.split()
        for word in words:
            synonyms = [
                syn.lemmas()[0].name() for syn in wordnet.synsets(word) if syn.lemmas()
            ]
            for syn in synonyms[:3]:
                new_kw = kw.replace(word, syn)
                variations.append(new_kw)
                if len(variations) >= num_variations:
                    return variations
    return variations[:num_variations]


def question_based_keywords(base_keywords, num_variations):
    """Generate question-based keyword variations."""
    # TODO: Customize prefixes based on industry or user intent
    prefixes = ["how to", "what is", "why does", "when to", "where to"]
    variations = []
    for kw in base_keywords:
        for prefix in prefixes:
            variations.append(f"{prefix} {kw}")
            if len(variations) >= num_variations:
                return variations
    return variations[:num_variations]


def semantic_clustering(base_keywords, num_variations):
    """Generate keyword variations using semantic clustering."""
    # TODO: Optimize cluster number dynamically using silhouette score
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(base_keywords)
    kmeans = KMeans(n_clusters=min(3, len(base_keywords)), random_state=42)
    clusters = kmeans.fit_predict(X)
    variations = []
    for i in range(min(3, len(base_keywords))):
        cluster_keywords = [
            kw for kw, cluster in zip(base_keywords, clusters) if cluster == i
        ]
        if len(cluster_keywords) > 1:
            variations.append(" ".join(cluster_keywords[:2]))
        if len(variations) >= num_variations:
            break
    return variations[:num_variations]


def modifier_addition(base_keywords, num_variations):
    """Add modifiers to generate keyword variations."""
    # TODO: Expand modifier list with user-configurable options
    modifiers = ["best", "top", "cheap", "easy", "durable", "quick", "effective"]
    variations = []
    for kw in base_keywords:
        for mod in modifiers:
            variations.append(f"{mod} {kw}")
            if len(variations) >= num_variations:
                return variations
    return variations[:num_variations]


def location_specific_keywords(base_keywords, num_variations):
    """Generate location-specific keyword variations."""
    # TODO: Integrate dynamic location list from user input or geolocation API
    locations = ["near me", "in New York", "in California", "online"]
    variations = []
    for kw in base_keywords:
        for loc in locations:
            variations.append(f"{kw} {loc}")
            if len(variations) >= num_variations:
                return variations
    return variations[:num_variations]


def seasonal_variations(base_keywords, num_variations):
    """Generate seasonal keyword variations."""
    # TODO: Add date-based logic to prioritize relevant seasons
    seasons = ["winter", "summer", "spring", "fall"]
    variations = []
    for kw in base_keywords:
        for season in seasons:
            variations.append(f"{season} {kw}")
            if len(variations) >= num_variations:
                return variations
    return variations[:num_variations]


def intent_driven_keywords(base_keywords, num_variations):
    """Generate intent-driven keyword variations."""
    # TODO: Refine intent detection with machine learning model
    intent_prefixes = {
        "transactional": "buy",
        "informational": "learn",
        "navigational": "find",
    }
    variations = []
    for kw in base_keywords:
        for intent, prefix in intent_prefixes.items():
            variations.append(f"{prefix} {kw}")
            if len(variations) >= num_variations:
                return variations
    return variations[:num_variations]


def word_combination(base_keywords, num_variations):
    """Combine base keywords with extra terms."""
    # TODO: Allow user-defined extra terms via configuration
    extra_terms = ["tips", "guide", "review", "essentials"]
    variations = []
    for kw in base_keywords:
        for term in extra_terms:
            variations.append(f"{kw} {term}")
            if len(variations) >= num_variations:
                return variations
    return variations[:num_variations]


def gemini_synonyms(base_keywords, num_variations):
    """Generate synonyms using Gemini model."""
    # TODO: Cache Gemini responses to reduce API calls
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Generate {num_variations} synonyms or related phrases for these keywords: {', '.join(base_keywords)}. Return in plain text, one per line."
    response = model.generate_content(prompt)
    variations = [v.strip() for v in response.text.strip().split("\n") if v.strip()]
    return variations[:num_variations]


def negative_keyword_avoidance(
    base_keywords, site_content_url, num_variations, respect_robots_txt=True
):
    """Generate variations avoiding common site terms."""
    # TODO: Improve term avoidance with semantic understanding
    site_content = (
        crawl_site_content(site_content_url, respect_robots_txt)
        if site_content_url
        else ""
    )
    if not site_content:
        return base_keywords[:num_variations]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform([site_content])
    common_terms = [
        term
        for term, score in sorted(
            zip(vectorizer.get_feature_names_out(), X.toarray()[0]),
            key=lambda x: x[1],
            reverse=True,
        )
    ][:5]
    variations = []
    for kw in base_keywords:
        if not any(term in kw for term in common_terms):
            variations.append(kw)
        else:
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = f"Generate a variation of '{kw}' avoiding these terms: {', '.join(common_terms)}. Return one phrase."
            response = model.generate_content(prompt)
            variations.append(response.text.strip())
        if len(variations) >= num_variations:
            break
    return variations[:num_variations]


@retry(
    stop_after_attempt(3),
    wait_exponential(multiplier=1, min=4, max=60),
    retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def generate_keyword_variations(
    base_keywords,
    site_url,
    num_variations,
    initial_data=None,
    min_words=2,
    max_words=5,
    num_results=10,
    use_synonyms=False,
    use_questions=False,
    use_clustering=False,
    use_modifiers=False,
    use_locations=False,
    use_seasons=False,
    use_intents=False,
    use_combinations=False,
    use_gemini_synonyms=False,
    use_negative_avoidance=False,
    site_content_url="",
    respect_robots_txt=True,
):
    """Generate keyword variations with multiple strategies."""
    # TODO: Add priority weighting for variation methods based on effectiveness
    logger.debug(f"Generating keyword variations for {base_keywords}")
    if not base_keywords:
        raise ValueError("No base keywords provided")

    variations = set()
    methods = [
        use_synonyms,
        use_questions,
        use_clustering,
        use_modifiers,
        use_locations,
        use_seasons,
        use_intents,
        use_combinations,
        use_gemini_synonyms,
        use_negative_avoidance,
    ]
    total_methods = sum(1 for m in methods if m) or 1
    per_method = max(1, num_variations // total_methods)

    model = genai.GenerativeModel("gemini-2.0-flash")
    gaps = analyze_competitor_gaps(base_keywords, site_url, num_results, respect_robots_txt)
    gap_prompt = "\nCompetitor gaps: " + ", ".join([f"{k}: {v}" for k, v in gaps.items()])
    prompt = (
        f"You are an SEO expert optimizing keywords for {site_url}. Given these existing keywords: {', '.join(base_keywords)}, "
        f"generate {per_method} new, high-quality, long-tail keyword variations that are semantically related but not redundant. "
        f"Focus on untapped opportunities based on these gaps: {gap_prompt}. Return in plain text, one per line."
    )
    response = model.generate_content(prompt)
    base_variations = [v.strip() for v in response.text.strip().split("\n") if v.strip()]
    variations.update(base_variations)

    if use_synonyms:
        variations.update(synonym_expansion(base_keywords, per_method))
    if use_questions:
        variations.update(question_based_keywords(base_keywords, per_method))
    if use_clustering:
        variations.update(semantic_clustering(base_keywords, per_method))
    if use_modifiers:
        variations.update(modifier_addition(base_keywords, per_method))
    if use_locations:
        variations.update(location_specific_keywords(base_keywords, per_method))
    if use_seasons:
        variations.update(seasonal_variations(base_keywords, per_method))
    if use_intents:
        variations.update(intent_driven_keywords(base_keywords, per_method))
    if use_combinations:
        variations.update(word_combination(base_keywords, per_method))
    if use_gemini_synonyms:
        variations.update(gemini_synonyms(base_keywords, per_method))
    if use_negative_avoidance and site_content_url:
        variations.update(
            negative_keyword_avoidance(
                base_keywords, site_content_url, per_method, respect_robots_txt
            )
        )

    if initial_data is not None:
        filtered_variations = filter_keyword_specificity(
            list(variations), initial_data, min_words, max_words
        )
        if not filtered_variations:
            logger.warning(
                "No keywords passed specificity filter; falling back to unfiltered variations"
            )
            filtered_variations = list(variations)
    else:
        filtered_variations = list(variations)

    if not filtered_variations:
        raise ValueError("No valid variations generated after fallback")
    logger.info(f"Generated {len(filtered_variations)} new keyword variations: {filtered_variations}")
    return list(set(filtered_variations))[:num_variations]


@retry(
    stop_after_attempt(3),
    wait_exponential(multiplier=1, min=4, max=60),
    retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def get_keyword_quality(keywords, site_url, intent_distribution=None):
    """Score keyword quality based on relevance, intent, and sentiment."""
    # TODO: Validate intent distribution against actual site content
    logger.debug(f"Scoring quality for {keywords[:5]}...")
    intent_prompt = f"Intent distribution: {intent_distribution}" if intent_distribution else ""
    prompt = (
        f"You are an SEO expert assessing keywords for {site_url}. For each keyword in the list below, provide a quality score (0-1) based on: "
        f"1) Relevance to site, 2) User intent clarity, 3) Positive sentiment potential. {intent_prompt}. "
        "Return the results in this exact format, one per line: 'keyword: score: intent: sentiment'. "
        "Example:\n"
        "'best skate shoes: 0.8: transactional: positive'\n"
        "'how to skate: 0.7: informational: neutral'\n\n"
        "Keywords to assess:\n" + "\n".join(keywords)
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    quality_scores = {}
    intents = {}
    sentiments = {}
    for line in lines:
        try:
            kw, score, intent, sentiment = line.split(":")
            quality_scores[kw.strip()] = float(score.strip())
            intents[kw.strip()] = intent.strip()
            sentiments[kw.strip()] = sentiment.strip()
        except ValueError:
            logger.warning(f"Invalid format in response line: {line}")
            continue

    if not quality_scores:
        logger.warning("No valid quality scores returned from Gemini; assigning defaults")
        for kw in keywords:
            quality_scores[kw] = 0.5
            intents[kw] = "navigational"
            sentiments[kw] = "neutral"

    logger.info(f"Scored quality for {len(quality_scores)} keywords")
    return quality_scores, intents, sentiments


def benchmark_against_historical(initial_data, new_keywords):
    """Benchmark new keywords against historical data."""
    # TODO: Add statistical significance testing for benchmarks
    logger.debug(f"Benchmarking {len(new_keywords)} keywords against historical data")
    vectorizer = TfidfVectorizer(stop_words="english")
    all_keywords = initial_data["keyword"].tolist() + new_keywords
    X = vectorizer.fit_transform(all_keywords)
    kmeans = KMeans(n_clusters=min(5, len(initial_data) // 2), random_state=42)
    clusters = kmeans.fit_predict(X)
    cluster_map = pd.DataFrame({"keyword": all_keywords, "cluster": clusters})

    cluster_map = cluster_map.drop_duplicates(subset="keyword", keep="first")

    historical_means = (
        initial_data.groupby("keyword")
        .agg({"ctr": "mean", "position": "mean"})
        .reset_index()
    )
    cluster_performance = (
        cluster_map.merge(historical_means, on="keyword", how="left")
        .groupby("cluster")
        .agg({"ctr": "mean", "position": "mean"})
        .fillna({"ctr": 0, "position": 100})
    )

    unique_new_keywords = list(dict.fromkeys(new_keywords))
    new_kw_df = pd.DataFrame({"keyword": unique_new_keywords})

    new_kw_df = (
        new_kw_df.merge(cluster_map, on="keyword", how="left")
        .merge(cluster_performance, on="cluster", how="left")
        .groupby("keyword")
        .agg({"ctr": "mean", "position": "mean"})
        .reset_index()
    )

    return new_kw_df.set_index("keyword")[["ctr", "position"]].to_dict("index")


def filter_keyword_specificity(keywords, initial_data, min_words, max_words):
    """Filter keywords by specificity and word count."""
    # TODO: Adjust specificity threshold dynamically based on corpus size
    logger.debug(f"Filtering {len(keywords)} keywords for specificity (min_words={min_words}, max_words={max_words})...")
    vectorizer = TfidfVectorizer(stop_words="english")
    initial_corpus = initial_data["keyword"].tolist()
    X_initial = vectorizer.fit_transform(initial_corpus)
    X_new = vectorizer.transform(keywords)
    specificity_scores = X_new.mean(axis=1).A1
    filtered = []
    for kw, score in zip(keywords, specificity_scores):
        word_count = len(kw.split())
        if min_words <= word_count <= max_words and score > 0.05:
            filtered.append(kw)
        logger.debug(f"Keyword: '{kw}', Word Count: {word_count}, Specificity Score: {score}")
    logger.info(f"Filtered to {len(filtered)} keywords")
    return filtered


@retry(
    stop_after_attempt(3),
    wait_exponential(multiplier=1, min=4, max=60),
    retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def get_x_social_validation(keywords):
    """Validate keywords using simulated X social data."""
    # TODO: Replace simulation with actual X API integration for real-time data
    # TODO: Add rate limiting handling specific to X API constraints
    logger.debug(f"Validating {keywords[:5]} via X posts...")
    social_scores = {}
    for kw in keywords:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = (
            f"Imagine 100 recent X posts about '{kw}'. "
            "Estimate mention volume (0-100) and sentiment (positive, negative, neutral)."
        )
        response = model.generate_content(prompt)
        try:
            volume = (
                min(100, max(0, int(response.text.split("volume:")[1].split()[0]))) / 100.0
            )
            sentiment = response.text.split("sentiment:")[1].strip().lower()
            sentiment_score = (
                1.0 if sentiment == "positive" else (0.5 if sentiment == "neutral" else 0.0)
            )
        except (IndexError, ValueError):
            volume, sentiment_score = 0, 0.5
        social_scores[kw] = {"volume": volume, "sentiment": sentiment_score}
        time.sleep(1)
    logger.info(f"Validated {len(social_scores)} keywords on X")
    return social_scores


def crawl_site_content(content_crawl_url, respect_robots_txt=True):
    """Crawl site content including linked pages."""
    # TODO: Implement depth limit for crawling to prevent infinite loops
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    site_content = ""
    try:
        if respect_robots_txt:
            rp = urllib.robotparser.RobotFileParser()
            base_url = f"{content_crawl_url.split('/')[0]}//{content_crawl_url.split('/')[2]}"
            rp.set_url(f"{base_url}/robots.txt")
            rp.read()
            if not rp.can_fetch(headers["User-Agent"], content_crawl_url):
                logger.warning(f"Blocked by robots.txt: {content_crawl_url}")
                return ""

        response = requests.get(content_crawl_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(
            [elem.get_text(strip=True) for elem in soup.find_all(["p", "h1", "h2", "h3"])]
        )
        site_content += text + " "

        links = [
            a.get("href")
            for a in soup.find_all("a", href=True)
            if a.get("href").startswith("/")
            or content_crawl_url.split("/")[2] in a.get("href")
        ]
        unique_links = list(set(links))[:5]
        for link in unique_links:
            full_url = (
                link
                if link.startswith("http")
                else f"{content_crawl_url.rstrip('/')}/{link.lstrip('/')}"
            )
            if respect_robots_txt and not rp.can_fetch(headers["User-Agent"], full_url):
                logger.warning(f"Blocked by robots.txt: {full_url}")
                continue
            try:
                link_response = requests.get(full_url, headers=headers, timeout=10)
                link_response.raise_for_status()
                link_soup = BeautifulSoup(link_response.text, "html.parser")
                link_text = " ".join(
                    [
                        elem.get_text(strip=True)
                        for elem in link_soup.find_all(["p", "h1", "h2", "h3"])
                    ]
                )
                site_content += link_text + " "
            except Exception as e:
                logger.warning(f"Failed to crawl {full_url}: {str(e)}")
            time.sleep(1)
    except Exception as e:
        logger.error(f"Failed to crawl {content_crawl_url}: {str(e)}")
        return ""

    return site_content[:10000]


def check_content_overlap(keywords, content_crawl_url, respect_robots_txt=True):
    """Check keyword overlap with site content."""
    # TODO: Use more advanced cosine similarity with contextual embeddings
    logger.debug(f"Checking overlap for {len(keywords)} keywords with site {content_crawl_url}")
    site_content_corpus = (
        crawl_site_content(content_crawl_url, respect_robots_txt)
        if content_crawl_url
        else ""
    )
    vectorizer = TfidfVectorizer(stop_words="english")
    if not site_content_corpus:
        return {kw: 0 for kw in keywords}
    X_corpus = vectorizer.fit_transform([site_content_corpus])
    X_keywords = vectorizer.transform(keywords)
    overlap_scores = cosine_similarity(X_keywords, X_corpus).flatten()
    return {kw: score for kw, score in zip(keywords, overlap_scores)}


def infer_dynamic_weights(initial_data):
    """Infer dynamic weights based on historical data metrics."""
    # TODO: Allow user override of inferred weights via configuration
    logger.debug("Inferring dynamic weights")
    avg_ctr = initial_data["ctr"].mean()
    avg_impressions = initial_data["impressions"].mean()
    avg_position = initial_data["position"].mean()
    if avg_ctr > 0.05:
        return {
            "volume": 0.2,
            "competition": 0.2,
            "ctr": 0.3,
            "position": 0.2,
            "quality": 0.1,
        }
    elif avg_impressions > 1000:
        return {
            "volume": 0.3,
            "competition": 0.15,
            "ctr": 0.15,
            "position": 0.2,
            "quality": 0.2,
        }
    elif avg_position < 10:
        return {
            "volume": 0.2,
            "competition": 0.2,
            "ctr": 0.2,
            "position": 0.3,
            "quality": 0.1,
        }
    return {
        "volume": 0.25,
        "competition": 0.2,
        "ctr": 0.2,
        "position": 0.2,
        "quality": 0.15,
    }


@retry(
    stop_after_attempt(3),
    wait_exponential(multiplier=1, min=4, max=60),
    retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def get_search_console_data(site_url, start_date, end_date, row_limit):
    """Fetch keyword data from Search Console."""
    # TODO: Cache Search Console data to reduce API calls
    logger.debug(f"Fetching Search Console data for {site_url}")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.join(root_dir, "token.json")
    creds = None

    if os.path.exists(token_path):
        logger.debug(f"Found existing token.json at {token_path}")
        try:
            with open(token_path, "r") as token_file:
                token_data = json.load(token_file)
            logger.debug(f"Raw token.json contents: {json.dumps(token_data, indent=2)}")

            creds = Credentials(
                token=token_data.get("token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=token_data.get("token_uri"),
                client_id=token_data.get("client_id"),
                client_secret=token_data.get("client_secret"),
                scopes=token_data.get("scopes"),
            )
            if "expiry" in token_data:
                creds.expiry = datetime.strptime(
                    token_data["expiry"], "%Y-%m-%dT%H:%M:%S.%fZ"
                )

            if creds.valid:
                logger.debug("Using valid existing access token")
            elif creds.expired and creds.refresh_token:
                logger.debug("Access token expired, refreshing with refresh_token")
                creds.refresh(Request())
                with open(token_path, "w") as token:
                    token.write(creds.to_json())
                logger.debug("Credentials refreshed and saved")
            else:
                logger.debug("Access token expired and no refresh_token available, regenerating")
                creds = None
        except Exception as e:
            logger.error(f"Error loading token.json: {str(e)}")
            creds = None

    if not creds:
        logger.debug("Generating new credentials with offline access")
        flow = InstalledAppFlow.from_client_secrets_file(
            SEARCH_CONSOLE_JSON_PATH,
            scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
            access_type="offline",
            prompt="consent",
        )
        creds = flow.run_local_server(port=5001)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
        logger.debug("New credentials generated and saved to token.json")

    service = build("searchconsole", "v1", credentials=creds)
    request = {
        "startDate": start_date,
        "endDate": end_date,
        "dimensions": ["query"],
        "rowLimit": row_limit,
    }
    response = service.searchanalytics().query(siteUrl=site_url, body=request).execute()
    keywords = []
    for row in response.get("rows", []):
        keyword = row["keys"][0]
        clicks = row.get("clicks", 0)
        impressions = row.get("impressions", 0)
        ctr = row.get("ctr", 0)
        position = row.get("position", 0)
        keywords.append(
            {
                "keyword": keyword,
                "clicks": clicks,
                "impressions": impressions,
                "ctr": ctr,
                "position": position,
            }
        )
    if not keywords:
        raise ValueError("No data returned from Search Console")
    logger.info(f"Extracted {len(keywords)} keywords")
    return pd.DataFrame(keywords)


@retry(
    stop_after_attempt(3),
    wait_exponential(multiplier=1, min=4, max=60),
    retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
)
def validate_keywords_with_planner(keywords):
    """Validate keywords using Google Ads Keyword Planner."""
    # TODO: Add regional targeting options for more precise validation
    logger.debug(f"Validating {len(keywords)} keywords with Google Ads")
    if not keywords:
        raise ValueError("No keywords to validate")
    client = GoogleAdsClient.load_from_storage(GOOGLE_ADS_YAML_PATH)
    keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
    BATCH_SIZE = 20
    validated = []
    for i in range(0, len(keywords), BATCH_SIZE):
        batch_keywords = keywords[i : i + BATCH_SIZE]
        request = client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = client.login_customer_id
        request.keyword_seed.keywords.extend(batch_keywords)
        historical_metrics_options = client.get_type("HistoricalMetricsOptions")
        start_year_month = client.get_type("YearMonth")
        start_year_month.year = 2023
        start_year_month.month = 12
        end_year_month = client.get_type("YearMonth")
        end_year_month.year = 2024
        end_year_month.month = 2
        year_month_range = client.get_type("YearMonthRange")
        year_month_range.start = start_year_month
        year_month_range.end = end_year_month
        historical_metrics_options.year_month_range = year_month_range
        request.historical_metrics_options = historical_metrics_options
        response = keyword_plan_idea_service.generate_keyword_ideas(request=request)
        response_keywords = {}
        for idea in response:
            keyword = idea.text
            if keyword in response_keywords:
                continue
            avg_monthly_searches = idea.keyword_idea_metrics.avg_monthly_searches
            competition = idea.keyword_idea_metrics.competition.name
            cpc = idea.keyword_idea_metrics.average_cpc_micros / 1_000_000
            comp_map = {"LOW": 0.33, "MEDIUM": 0.66, "HIGH": 1.0}
            competition_index = comp_map.get(competition, 0.66)
            response_keywords[keyword] = {
                "search_volume": avg_monthly_searches,
                "competition": competition_index,
                "cpc": cpc,
                "source": "Google Ads Idea",
            }
        for kw in batch_keywords:
            if kw in response_keywords:
                data = response_keywords[kw]
                validated.append(
                    {
                        "keyword": kw,
                        "search_volume": data["search_volume"],
                        "competition": data["competition"],
                        "cpc": data["cpc"],
                        "source": "Gemini",
                    }
                )
            else:
                validated.append(
                    {
                        "keyword": kw,
                        "search_volume": 0,
                        "competition": 1.0,
                        "cpc": 0,
                        "source": "Gemini",
                    }
                )
        for kw, data in response_keywords.items():
            if kw not in batch_keywords:
                validated.append(
                    {
                        "keyword": kw,
                        "search_volume": data["search_volume"],
                        "competition": data["competition"],
                        "cpc": data["cpc"],
                        "source": "Google Ads Idea",
                    }
                )
    batch_count = (len(keywords) + BATCH_SIZE - 1) // BATCH_SIZE
    if not validated:
        raise ValueError("No valid keyword data returned")
    logger.info(f"Validated {len(validated)} keywords in {batch_count} batches")
    return pd.DataFrame(validated), batch_count


def refine_keywords(
    initial_data,
    site_url,
    iterations,
    weights,
    base_keywords_per_iteration,
    output_limit,
    content_crawl_url="",
    num_variations=10,
    min_words=2,
    max_words=5,
    num_results=10,
    use_synonyms=False,
    use_questions=False,
    use_clustering=False,
    use_modifiers=False,
    use_locations=False,
    use_seasons=False,
    use_intents=False,
    use_combinations=False,
    use_gemini_synonyms=False,
    use_negative_avoidance=False,
    respect_robots_txt=True,
):
    """Refine keywords through multiple iterations."""
    # TODO: Implement early stopping if improvement plateaus
    logger.debug(f"Refining keywords with iterations={iterations}")
    if initial_data.empty:
        raise ValueError("Initial Search Console data is empty")
    original_data = initial_data.copy()
    original_data["source"] = "Search Console"
    all_data = original_data.copy()
    current_keywords = all_data["keyword"].tolist()
    total_batch_count = 0
    intent_dist = initial_data["intent"].value_counts(normalize=True).to_dict()

    for i in range(iterations):
        logger.debug(f"Iteration {i+1}/{iterations}")
        new_keywords = generate_keyword_variations(
            current_keywords[:base_keywords_per_iteration],
            site_url,
            num_variations,
            initial_data,
            min_words,
            max_words,
            num_results,
            use_synonyms,
            use_questions,
            use_clustering,
            use_modifiers,
            use_locations,
            use_seasons,
            use_intents,
            use_combinations,
            use_gemini_synonyms,
            use_negative_avoidance,
            content_crawl_url,
            respect_robots_txt,
        )
        validated_data, batch_count = validate_keywords_with_planner(new_keywords)
        total_batch_count += batch_count

        trends_scores = get_trends_score(validated_data["keyword"].tolist(), initial_data)
        quality_scores, intents, sentiments = get_keyword_quality(
            validated_data["keyword"].tolist(), site_url, intent_dist
        )
        historical_benchmarks = benchmark_against_historical(
            initial_data, validated_data["keyword"].tolist()
        )
        social_scores = get_x_social_validation(validated_data["keyword"].tolist())
        overlap_scores = check_content_overlap(
            validated_data["keyword"].tolist(), content_crawl_url, respect_robots_txt
        )

        validated_data["source"] = "New Gemini Suggestion"
        validated_data["trends_score"] = validated_data["keyword"].map(
            lambda x: trends_scores.get(x, 0) / 100
        )
        validated_data["quality"] = validated_data["keyword"].map(
            lambda x: quality_scores.get(x, 0.5)
        )
        validated_data["intent"] = validated_data["keyword"].map(
            lambda x: intents.get(x, "navigational")
        )
        validated_data["sentiment"] = validated_data["keyword"].map(
            lambda x: 1 if sentiments.get(x, "positive") == "positive" else 0.5
        )
        validated_data["hist_ctr"] = validated_data["keyword"].map(
            lambda x: historical_benchmarks.get(x, {}).get("ctr", 0)
        )
        validated_data["hist_position"] = validated_data["keyword"].map(
            lambda x: historical_benchmarks.get(x, {}).get("position", 100)
        )
        validated_data["social_volume"] = validated_data["keyword"].map(
            lambda x: social_scores.get(x, {}).get("volume", 0)
        )
        validated_data["social_sentiment"] = validated_data["keyword"].map(
            lambda x: social_scores.get(x, {}).get("sentiment", 0.5)
        )
        validated_data["overlap"] = validated_data["keyword"].map(
            lambda x: overlap_scores.get(x, 0)
        )

        all_data = pd.concat([all_data, validated_data], ignore_index=True).drop_duplicates(
            subset=["keyword"]
        )

        all_data["search_volume"] = all_data["search_volume"].fillna(0)
        all_data["competition"] = all_data["competition"].fillna(1.0)
        all_data["ctr"] = all_data["ctr"].fillna(0)
        all_data["position"] = all_data["position"].fillna(100)
        all_data["quality"] = all_data["quality"].fillna(0.5)
        all_data["trends_score"] = all_data["trends_score"].fillna(0)
        all_data["hist_ctr"] = all_data["hist_ctr"].fillna(0)
        all_data["hist_position"] = all_data["hist_position"].fillna(100)
        all_data["social_volume"] = all_data["social_volume"].fillna(0)
        all_data["social_sentiment"] = all_data["social_sentiment"].fillna(0.5)
        all_data["overlap"] = all_data["overlap"].fillna(0)

        max_volume = all_data["search_volume"].max() if all_data["search_volume"].max() > 0 else 1
        all_data["norm_volume"] = all_data["search_volume"] / max_volume
        all_data["norm_competition"] = 1 - (
            all_data["competition"] / all_data["competition"].max()
        )
        all_data["norm_ctr"] = all_data["ctr"] / max(all_data["ctr"].max(), 1)
        all_data["norm_position"] = 1 - (
            all_data["position"] / max(all_data["position"].max(), 100)
        )
        all_data["norm_quality"] = all_data["quality"]
        all_data["norm_trends"] = all_data["trends_score"]
        all_data["norm_hist_ctr"] = all_data["hist_ctr"] / max(all_data["hist_ctr"].max(), 1)
        all_data["norm_hist_position"] = 1 - (all_data["hist_position"] / 100)
        all_data["norm_social_volume"] = all_data["social_volume"]
        all_data["norm_social_sentiment"] = all_data["social_sentiment"]
        all_data["norm_overlap"] = 1 - all_data["overlap"]

        all_data["score"] = (
            weights["volume"] * all_data["norm_volume"]
            + weights["competition"] * all_data["norm_competition"]
            + weights["ctr"] * all_data["norm_ctr"]
            + weights["position"] * all_data["norm_position"]
            + weights["quality"] * all_data["norm_quality"]
            + 0.1 * all_data["norm_trends"]
            + 0.1 * all_data["norm_hist_ctr"]
            + 0.1 * all_data["norm_hist_position"]
            + 0.05 * all_data["norm_social_volume"]
            + 0.05 * all_data["norm_social_sentiment"]
            + 0.1 * all_data["norm_overlap"]
        )

        all_data = all_data.sort_values("score", ascending=False)
        current_keywords = all_data["keyword"].tolist()
        logger.debug(f"Iteration {i+1} complete, total keywords: {len(all_data)}")

    reward_threshold = all_data["score"].quantile(0.75)
    all_data["reward"] = all_data["score"].apply(
        lambda x: "Yes" if x >= reward_threshold else "No"
    )
    logger.info(f"Refinement complete: {len(all_data)} keywords")
    return all_data, total_batch_count


def cluster_keywords(df, max_clusters):
    """Cluster keywords using KMeans."""
    # TODO: Explore alternative clustering algorithms (e.g., DBSCAN) for better results
    logger.debug(f"Clustering {len(df)} keywords")
    if len(df) < 2:
        df["cluster"] = 0
        return df
    n_clusters = min(max_clusters, len(df) // 2)
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["keyword"])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)
    logger.info(f"Clustered into {n_clusters} groups")
    return df


def categorize_intent(keyword):
    """Categorize keyword intent."""
    # TODO: Enhance intent detection with NLP model for more accuracy
    keyword = keyword.lower()
    if any(x in keyword for x in ["buy", "price", "order"]):
        return "transactional"
    elif any(x in keyword for x in ["how to", "guide", "what is"]):
        return "informational"
    else:
        return "navigational"


def export_results(df, filename):
    """Export results to CSV and JSON."""
    # TODO: Add option for additional export formats (e.g., Excel)
    logger.debug(f"Exporting results for {filename}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{filename}_{timestamp}"
    csv_path = Path(RESULTS_FOLDER) / f"{unique_filename}.csv"
    json_path = Path(RESULTS_FOLDER) / f"{unique_filename}.json"
    export_columns = [
        "keyword",
        "search_volume",
        "competition",
        "ctr",
        "position",
        "quality",
        "score",
        "reward",
        "source",
        "cluster",
        "intent",
        "trends_score",
        "hist_ctr",
        "hist_position",
        "social_volume",
        "social_sentiment",
        "overlap",
    ]
    export_df = df[[col for col in export_columns if col in df.columns]]
    export_df.to_csv(csv_path, index=False)
    export_df.to_json(json_path, orient="records")
    logger.info(f"Exported to {csv_path} and {json_path}")
    return unique_filename, csv_path, json_path


@app.route("/", methods=["GET"])
def index():
    """Render the index page with default date range."""
    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    session.pop("results_key", None)
    return render_template(
        "index.html", default_start=default_start, default_end=default_end, form_data=None
    )


@app.route("/process", methods=["POST", "GET"])
def process_keywords():
    """Process keywords and display results with pagination."""
    # TODO: Add progress feedback for long-running processes
    page = request.args.get("page", 1, type=int)
    ROWS_PER_PAGE = 10

    if request.method == "POST":
        configure_logging(request.form.get("enable_logging") == "on")
        logger.debug("Processing keywords (POST)")

        site_url = request.form.get("site_url")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        row_limit = request.form.get("row_limit", "50")
        iterations = request.form.get("iterations", "3")
        weight_volume = request.form.get("weight_volume", "30")
        weight_competition = request.form.get("weight_competition", "20")
        weight_ctr = request.form.get("weight_ctr", "20")
        weight_position = request.form.get("weight_position", "20")
        weight_quality = request.form.get("weight_quality", "10")
        base_keywords_per_iteration = request.form.get("base_keywords_per_iteration", "10")
        output_limit = request.form.get("output_limit", "50")
        max_clusters = request.form.get("max_clusters", "3")
        num_variations = request.form.get("num_variations", "10")
        min_words = request.form.get("min_words", "2")
        max_words = request.form.get("max_words", "5")
        num_results = request.form.get("num_results", "10")
        content_crawl_url = request.form.get("content_crawl_url", "")
        use_synonyms = request.form.get("use_synonyms") == "on"
        use_questions = request.form.get("use_questions") == "on"
        use_clustering = request.form.get("use_clustering") == "on"
        use_modifiers = request.form.get("use_modifiers") == "on"
        use_locations = request.form.get("use_locations") == "on"
        use_seasons = request.form.get("use_seasons") == "on"
        use_intents = request.form.get("use_intents") == "on"
        use_combinations = request.form.get("use_combinations") == "on"
        use_gemini_synonyms = request.form.get("use_gemini_synonyms") == "on"
        use_negative_avoidance = request.form.get("use_negative_avoidance") == "on"
        respect_robots_txt = request.form.get("respect_robots_txt") == "on"

        errors = []
        required_fields = {
            "site_url": site_url,
            "start_date": start_date,
            "end_date": end_date,
            "row_limit": row_limit,
            "iterations": iterations,
            "weight_volume": weight_volume,
            "weight_competition": weight_competition,
            "weight_ctr": weight_ctr,
            "weight_position": weight_position,
            "weight_quality": weight_quality,
            "base_keywords_per_iteration": base_keywords_per_iteration,
            "output_limit": output_limit,
            "max_clusters": max_clusters,
            "num_variations": num_variations,
            "min_words": min_words,
            "max_words": max_words,
            "num_results": num_results,
        }
        for field_name, field_value in required_fields.items():
            if not field_value:
                errors.append(f"Missing required field: {field_name}")

        if not errors:
            try:
                row_limit = int(row_limit)
                iterations = int(iterations)
                base_keywords_per_iteration = int(base_keywords_per_iteration)
                output_limit = int(output_limit)
                max_clusters = int(max_clusters)
                num_variations = int(num_variations)
                min_words = int(min_words)
                max_words = int(max_words)
                num_results = int(num_results)

                initial_data_temp = get_search_console_data(
                    site_url, start_date, end_date, row_limit
                )
                weights = infer_dynamic_weights(initial_data_temp)
                if all(
                    [
                        weight_volume,
                        weight_competition,
                        weight_ctr,
                        weight_position,
                        weight_quality,
                    ]
                ):
                    weights = {
                        "volume": float(weight_volume) / 100,
                        "competition": float(weight_competition) / 100,
                        "ctr": float(weight_ctr) / 100,
                        "position": float(weight_position) / 100,
                        "quality": float(weight_quality) / 100,
                    }
                total_weight = sum(weights.values())
                if not 0.99 <= total_weight <= 1.01:
                    errors.append("Weights must sum to 100% (excluding additional metrics)")
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                if (
                    start_dt > end_dt
                    or end_dt > datetime.now()
                    or row_limit < 1
                    or row_limit > 1000
                    or iterations < 1
                    or iterations > 10
                    or base_keywords_per_iteration < 1
                    or base_keywords_per_iteration > 20
                    or output_limit < 1
                    or output_limit > 200
                    or max_clusters < 1
                    or max_clusters > 10
                    or num_variations < 1
                    or num_variations > 50
                    or min_words < 1
                    or max_words < min_words
                    or num_results < 1
                    or num_results > 10
                ):
                    errors.append("Invalid input values (check ranges: num_results 1-10, etc.)")
            except ValueError as e:
                errors.append(f"Invalid input format: {str(e)}")
            except Exception as e:
                errors.append(f"Error during initial setup: {str(e)}")

        if errors:
            default_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            default_end = datetime.now().strftime("%Y-%m-%d")
            return render_template(
                "index.html",
                errors=errors,
                default_start=start_date or default_start,
                default_end=end_date or default_end,
                form_data=request.form,
            )

        genai.configure(api_key=GEMINI_API_KEY)

        try:
            initial_data = get_search_console_data(site_url, start_date, end_date, row_limit)
            initial_data["intent"] = initial_data["keyword"].apply(categorize_intent)
            refined_data, total_batch_count = refine_keywords(
                initial_data,
                site_url,
                iterations,
                weights,
                base_keywords_per_iteration,
                output_limit,
                content_crawl_url,
                num_variations,
                min_words,
                max_words,
                num_results,
                use_synonyms,
                use_questions,
                use_clustering,
                use_modifiers,
                use_locations,
                use_seasons,
                use_intents,
                use_combinations,
                use_gemini_synonyms,
                use_negative_avoidance,
                respect_robots_txt,
            )
            new_keywords_count = len(
                set(refined_data["keyword"]) - set(initial_data["keyword"])
            )
            clustered_data = cluster_keywords(refined_data, max_clusters)
            clustered_data["intent"] = clustered_data["keyword"].apply(categorize_intent)
            base_filename = f"keyword_report_{site_url.replace('https://', '').replace('/', '')}"
            unique_filename, csv_path, json_path = export_results(clustered_data, base_filename)

            results_key = str(uuid.uuid4())
            session["results_key"] = results_key
            session["metadata"] = {
                "site_url": site_url,
                "iterations": iterations,
                "base_keywords_per_iteration": base_keywords_per_iteration,
                "output_limit": output_limit,
                "max_clusters": max_clusters,
                "num_variations": num_variations,
                "min_words": min_words,
                "max_words": max_words,
                "num_results": num_results,
                "total_batch_count": total_batch_count,
                "new_keywords_count": new_keywords_count,
                "csv_file": f"{unique_filename}.csv",
                "json_file": f"{unique_filename}.json",
            }
            temp_data_path = Path(RESULTS_FOLDER) / f"{results_key}.json"
            clustered_data.to_json(temp_data_path, orient="records")
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            errors.append(f"Processing error: {str(e)}")
            return render_template(
                "index.html",
                errors=errors,
                default_start=start_date,
                default_end=end_date,
                form_data=request.form,
            )

    if "results_key" not in session:
        return render_template(
            "index.html", errors=["No results available. Submit the form first."]
        )

    results_key = session["results_key"]
    metadata = session["metadata"]
    temp_data_path = Path(RESULTS_FOLDER) / f"{results_key}.json"
    if not temp_data_path.exists():
        return render_template("index.html", errors=["Results data expired or missing."])

    clustered_data = pd.read_json(temp_data_path)
    sample_columns = [
        "keyword",
        "search_volume",
        "competition",
        "ctr",
        "position",
        "quality",
        "score",
        "reward",
        "source",
        "cluster",
        "intent",
        "trends_score",
        "hist_ctr",
        "hist_position",
        "social_volume",
        "social_sentiment",
        "overlap",
    ]
    full_data = clustered_data[sample_columns].head(int(metadata["output_limit"]))
    total_rows = len(full_data)
    total_pages = (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * ROWS_PER_PAGE
    end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
    paginated_data = full_data.iloc[start_idx:end_idx]

    full_table = paginated_data.to_html(
        classes="table table-striped",
        index=False,
        header=True,
        escape=False,
        formatters={
            "reward": lambda x: '<span class="reward-star">⭐</span>' if x == "Yes" else ""
        },
        table_id="results-table",
    ).replace('th>', 'th data-bs-toggle="tooltip" data-bs-placement="top"').replace(
        "keyword>", 'keyword title="The keyword phrase">'
    ).replace(
        "search_volume>", 'search_volume title="Avg monthly searches from Google Ads">'
    ).replace(
        "competition>", 'competition title="Competition level (0-1, lower is better)">'
    ).replace(
        "ctr>", 'ctr title="Click-through rate from Search Console">'
    ).replace(
        "position>", 'position title="Avg position in search results">'
    ).replace(
        "quality>", 'quality title="Gemini quality score (0-1)">'
    ).replace(
        "score>", 'score title="Overall weighted score">'
    ).replace(
        "reward>", 'reward title="Top 25% by score marked with ⭐">'
    ).replace(
        "source>", 'source title="Origin of the keyword">'
    ).replace(
        "cluster>", 'cluster title="Cluster group ID">'
    ).replace(
        "intent>", 'intent title="User intent (informational, transactional, navigational)">'
    ).replace(
        "trends_score>", 'trends_score title="Local trend score based on impressions (0-1)">'
    ).replace(
        "hist_ctr>", 'hist_ctr title="Historical CTR from similar keywords">'
    ).replace(
        "hist_position>", 'hist_position title="Historical avg position from similar keywords">'
    ).replace(
        "social_volume>", 'social_volume title="Social mention volume from X (0-1)">'
    ).replace(
        "social_sentiment>", 'social_sentiment title="Social sentiment score from X (0-1)">'
    ).replace(
        "overlap>", 'overlap title="Similarity to existing site content (0-1, lower is better)">'
    )

    return render_template(
        "results.html",
        full_table=full_table,
        csv_file=metadata["csv_file"],
        json_file=metadata["json_file"],
        site_url=metadata["site_url"],
        iterations=metadata["iterations"],
        base_keywords_per_iteration=metadata["base_keywords_per_iteration"],
        output_limit=metadata["output_limit"],
        max_clusters=metadata["max_clusters"],
        total_batch_count=metadata["total_batch_count"],
        new_keywords_count=metadata["new_keywords_count"],
        page=page,
        total_pages=total_pages,
    )


@app.route("/results/<filename>")
def download_file(filename):
    """Serve file for download from results folder."""
    file_path = Path(RESULTS_FOLDER) / filename
    if not file_path.exists():
        abort(404)
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    """Start the Flask application in debug mode."""
    logger.info("Starting Flask application")
    app.run(debug=True)